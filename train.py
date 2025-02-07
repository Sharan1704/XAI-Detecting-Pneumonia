import tensorflow as tf
from model import create_model
from data_preprocessing import create_data_generators

def train_model(train_dir, val_dir, test_dir):
    # Create data generators
    train_generator, val_generator, test_generator = create_data_generators(
        train_dir, val_dir, test_dir
    )
    
    # Create model
    model = create_model()
    
    # Model training
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_steps=val_generator.samples // val_generator.batch_size
    )
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    
    # Save model
    model.save('pneumonia_classification_model.h5')
    
    return model, history

# Main execution
if __name__ == '__main__':
    train_dir = r'E:\Medical Image Classifier\Data\chest_xray\train'
    val_dir = r'E:\Medical Image Classifier\Data\chest_xray\val'
    test_dir = r'E:\Medical Image Classifier\Data\chest_xray\test'
    
    model, training_history = train_model(train_dir, val_dir, test_dir)