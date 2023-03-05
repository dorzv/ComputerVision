import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from dataset import rev_label_map, label_color_map, label_map
from ssd_network import SSD

device = "cuda" if torch.cuda.is_available() else "cpu"

num_classes = len(label_map)
checkpoint = 'checkpoint_ssd300.pt'
checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
print(f'\nLoaded checkpoint from epoch {start_epoch}.\n')
model = SSD(num_classes=num_classes)
model.load_state_dict(checkpoint['model'])
model = model.to(device)
model.eval()

# Transformations
trans = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def detect(original_image, min_score, max_overlap, top_k):
    """
    A function to detect objects in image
    Args:
        original_image (PIL.Image): the image to detect objects in
        min_score (float): the minimum score a box needs to considered as detection of a class
        max_overlap (float): the maximum overlap two boxes can have and still considered as two separate
            detection. If the overlap is greater than this value, the box with the lower score will be suppressed.
        top_k (int): keep only the k boxes with the highest scores

    Returns:
        (PIL.Image): the original image with annotations for the objects detected
    """

    # Transform and move to device
    image = trans(original_image).to(device)

    # Forward propagation
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    detected_boxes, detected_labels, detected_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                                       max_overlap=max_overlap, top_k=top_k)

    # Move detections to the CPU
    detected_boxes = detected_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.tensor(2*[original_image.width, original_image.height], dtype=torch.float).unsqueeze(0)
    detected_boxes = detected_boxes * original_dims

    # Decode class integer value to labels
    detected_labels = [rev_label_map[label] for label in detected_labels[0].to('cpu').tolist()]

    # If no objects found, return the original image without annotations
    if detected_labels == ['background']:
        return original_image

    # Add anotations for the objects founded
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype('FreeMono.ttf', 15)

    for ii in range(detected_boxes.shape[0]):

        # Draw boxes
        box_location = detected_boxes[ii].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[detected_labels[ii]])
        draw.rectangle(xy=[loc + 1 for loc in box_location], outline=label_color_map[detected_labels[ii]])

        # Draw text
        text_size = font.getbbox(detected_labels[ii].upper())[2:]
        text_location = [box_location[0] + 2, box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4, box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[detected_labels[ii]])
        draw.text(xy=text_location, text=detected_labels[ii].upper(), fill='gray', font=font)

    del draw

    return annotated_image


if __name__ == '__main__':
    img_path = './Data/VOC2007/JPEGImages/000001.jpg'
    original_image = Image.open(img_path, mode='r')
    original_image = original_image.convert('RGB')
    detect(original_image, min_score=0.2, max_overlap=0.5, top_k=200).show()