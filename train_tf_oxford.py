"""
用你之前的 TensorFlow CNN 训练 Oxford-IIIT Pet（1.2 规定数据集）
- 数据：先运行 export_oxford_for_tf.py 导出为按类分文件夹，再由此脚本加载
- 模型：从零训练的小型 CNN（无预训练），Rescaling + 卷积 + 全连接
- 训练：30 epochs，trainval 已拆为 train/val；test 仅最后评估一次
"""
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# 路径：与 export_oxford_for_tf.py 一致
DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")
OXFORD_FOLDERS = os.path.join(DATA_ROOT, "oxford_pet_folders")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
NUM_CLASSES = 37
EPOCHS = 30
VALIDATION_SPLIT = 0.0  # 已用 export 时拆好 train/val

train_dir = os.path.join(OXFORD_FOLDERS, "train")
val_dir = os.path.join(OXFORD_FOLDERS, "val")
test_dir = os.path.join(OXFORD_FOLDERS, "test")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pet_classifier_tf.keras")


def load_train_val():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
        label_mode="int",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        label_mode="int",
    )
    return train_ds, val_ds


def load_test():
    return tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        label_mode="int",
    )


def get_class_weights(train_ds):
    labels = np.concatenate([y.numpy() for _, y in train_ds], axis=0)
    try:
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(labels)
        weights = compute_class_weight("balanced", classes=classes, y=labels)
        return dict(zip(classes, weights))
    except Exception:
        return None  # 无 sklearn 则不使用 class_weight


def main():
    if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
        print("请先运行: python export_oxford_for_tf.py")
        return

    train_ds, val_ds = load_train_val()
    class_names = train_ds.class_names
    print("Classes:", class_names, "->", len(class_names))
    class_weights = get_class_weights(train_ds)
    if class_weights:
        print("Class weights (sample):", list(class_weights.items())[:5])
    else:
        print("Class weights: 未使用（需 scikit-learn）")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    # 数据增强（仅训练集，符合作业 1.2）
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.3),
    ])

    # 从零训练的 CNN（无预训练）- 加深加宽以缓解欠拟合
    model = models.Sequential([
        data_augmentation,
        layers.Rescaling(1.0 / 255, input_shape=(*IMAGE_SIZE, 3)),
        layers.Conv2D(32, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.15),

        layers.Conv2D(64, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        layers.Conv2D(128, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(256, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.GlobalAveragePooling2D(),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            save_best_only=True,
            monitor="val_accuracy",
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=6,
            restore_best_weights=True,
            monitor="val_loss",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
    ]

    print("开始训练（共 30 个 epoch，早停 patience=6）...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights if class_weights else None,
    )

    # 仅最后用 test 集评估一次（符合作业 1.2）
    print("\n--- 仅在 test 集上做最终评估 ---")
    test_ds = load_test()
    test_ds = test_ds.cache().prefetch(AUTOTUNE)
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test Loss: {test_loss:.4f}  Test Acc: {test_acc:.4f}")

    # 保存类别名供推理用
    with open(os.path.join(os.path.dirname(__file__), "class_names_tf.json"), "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    print("最佳模型已保存:", MODEL_PATH)


if __name__ == "__main__":
    main()
