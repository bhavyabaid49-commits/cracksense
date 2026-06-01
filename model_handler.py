import torch
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
from PIL import Image

def load_crack_model():
    # 1. Load a pre-trained ResNet18 base architecture
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # 2. Reconfigure the output layer for your 3 specific classes
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 3) 
    
    
    # 3. Freeze the layers completely to evaluation mode
    model.eval()
    return model

# 🔥 PERSISTENT GLOBAL LOADING: This loads the model ONCE when the server boots up.
# This prevents random weight re-initialization on subsequent scans.
GLOBAL_CRACK_MODEL = load_crack_model()

def predict_image_severity(image_path):
    """
    Takes a local image path, processes the image using OpenCV, 
    and passes it to our persistently loaded global ResNet AI model.
    """
    try:
        # 1. Load the image using OpenCV 
        img = cv2.imread(image_path)
        if img is None:
            return {"error": "Could not read the image file."}
        
        # 2. Convert color from BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # 3. Preprocess image matrix targets
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        input_tensor = preprocess(pil_img)
        input_batch = input_tensor.unsqueeze(0) # Add a batch dimension
        
        # 4. Reference the stable GLOBAL model instead of spinning up a new one
        with torch.no_grad():
            output = GLOBAL_CRACK_MODEL(input_batch)
        
        # 5. Convert the scores into probabilities
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # Define our crack severity classes 
        classes = ["hairline", "moderate", "critical"]
        
        # Find which class got the highest score
        highest_score_index = torch.argmax(probabilities).item()
        detected_severity = classes[highest_score_index]
        confidence = float(probabilities[highest_score_index])
        
        return {
            "severity": detected_severity,
            "confidence": round(confidence, 2)
        }
        
    except Exception as e:
        return {"error": str(e)}