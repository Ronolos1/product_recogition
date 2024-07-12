# train_model.py
from data_preparation import prepare_data
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def train_model(train_generator, val_generator, initial_epochs=10, fine_tune_epochs=10, save_model_path='model_finetuned.keras'):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(train_generator.num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model with the base model frozen
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    initial_history = model.fit(train_generator, validation_data=val_generator, epochs=initial_epochs)

    # Gradually unfreeze layers and fine-tune the model
    num_layers_to_unfreeze = len(base_model.layers) // 5
    for i in range(1, 6):
        for layer in base_model.layers[-i * num_layers_to_unfreeze:]:
            layer.trainable = True

        model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
        fine_tune_history = model.fit(train_generator, validation_data=val_generator, epochs=fine_tune_epochs)

    model.save(save_model_path, save_format='keras')
    return model, initial_history, fine_tune_history

if __name__ == "__main__":
    train_generator, val_generator = prepare_data()
    model, initial_history, fine_tune_history = train_model(train_generator, val_generator, initial_epochs=10, fine_tune_epochs=10)
