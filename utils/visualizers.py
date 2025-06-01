import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns


def visualize_predictions(X_test, y_test, model, class_names, model_type='rf', 
                          num_images=25, class_indices=None, original_class_ids=None):
    if model_type == 'rf' or model_type == 'random_forest':
        y_pred_proba = model.predict_proba(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        reshape_dim = (32, 32, 3)
    elif model_type == 'vgg16':
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        for i in range(min(3, len(y_pred_proba))):
            top_indices = np.argsort(y_pred_proba[i])[-3:][::-1]
            top_probs = y_pred_proba[i][top_indices]
            for idx, prob in zip(top_indices, top_probs):
                if class_indices:
                    class_indices_inv = {v: k for k, v in class_indices.items()}
                    if idx in class_indices_inv:
                        class_id_str = class_indices_inv[idx]
                        class_id = int(class_id_str)
                        if isinstance(class_names, dict):
                            class_name = class_names.get(class_id, f"Unknown class {class_id}")
                        else:
                            class_name = class_names[class_id] if class_id < len(class_names) else f"Unknown class {class_id}"
                else:
                    if isinstance(class_names, dict):
                        class_name = class_names.get(idx, f"Unknown class {idx}")
                    else:
                        class_name = class_names[idx] if idx < len(class_names) else f"Unknown class {idx}"
    else:
        raise ValueError(f"Error: '{model_type}'")
    num_display = min(num_images, len(X_test))
    indices = list(range(num_display))
    n_cols = 5
    n_rows = (num_display + n_cols - 1) // n_cols
    plt.figure(figsize=(20, 4 * n_rows))
    class_indices_inv = None
    if class_indices:
        class_indices_inv = {v: int(k) for k, v in class_indices.items()}
    for i, idx in enumerate(indices):
        plt.subplot(n_rows, n_cols, i + 1)
        if model_type == 'rf' or model_type == 'random_forest':
            img = X_test[idx].reshape(reshape_dim)
            if img.max() <= 1.0:
                img = img * 255
        elif model_type == 'vgg16':
            img = X_test[idx].copy()
            img = (img - img.min()) / (img.max() - img.min()) * 255.0
        
        img = img.astype(np.uint8)
        plt.imshow(img)
        plt.axis('off')
        if model_type == 'vgg16':
            true_class_id = int(original_class_ids[idx])
            pred_idx = int(y_pred[idx])
            if class_indices_inv and pred_idx in class_indices_inv:
                pred_class_id = class_indices_inv[pred_idx]
            else:
                pred_class_id = pred_idx
            if isinstance(class_names, dict):
                true_name = class_names.get(true_class_id, f"Unknown class {true_class_id}")
                pred_name = class_names.get(pred_class_id, f"Unknown class {pred_class_id}")
            else:
                true_name = class_names[true_class_id] if true_class_id < len(class_names) else f"Unknown class {true_class_id}"
                pred_name = class_names[pred_class_id] if pred_class_id < len(class_names) else f"Unknown class {pred_class_id}"
            color = 'green' if true_class_id == pred_class_id else 'red'
                
        else:
            true_class = int(y_test[idx])
            pred_class = int(y_pred[idx])
            if isinstance(class_names, dict):
                true_name = class_names.get(true_class, f"Unknown class {true_class}")
                pred_name = class_names.get(pred_class, f"Unknown class {pred_class}")
            else:
                true_name = class_names[true_class] if true_class < len(class_names) else f"Unknown class {true_class}"
                pred_name = class_names[pred_class] if pred_class < len(class_names) else f"Unknown class {pred_class}"
            color = 'green' if true_class == pred_class else 'red'
        plt.title(f"Actual: {true_name}\nPredict: {pred_name}",
                  color=color, fontsize=10)

    plt.tight_layout()
    plt.suptitle(f"{model_type.upper()} Model Predictions", fontsize=16)
    plt.subplots_adjust(top=0.92)
    plt.show()
