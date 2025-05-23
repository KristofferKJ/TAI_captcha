import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


# Step 1: Define the alphabet and the label converter
class LabelConverter:
    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.char2idx = {char: i + 1 for i, char in enumerate(alphabet)}
        self.idx2char = {i + 1: char for i, char in enumerate(alphabet)}
        self.blank = 0  # CTC requires a blank token at index 0

    def encode(self, texts):
        lengths = [len(text) for text in texts]
        targets = [self.char2idx[char] for text in texts for char in text]
        return torch.tensor(targets, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)

    # Simplified decode method without verbose logging
    def decode(self, preds, pred_lengths):
        preds = preds.argmax(2)
        results = []
        for i in range(preds.shape[1]):
            seq = preds[:, i]
            chars = []
            prev = self.blank
            
            for j in range(pred_lengths[i]):
                if seq[j] != prev and seq[j] != self.blank:
                    if seq[j].item() in self.idx2char:
                        chars.append(self.idx2char[seq[j].item()])
                prev = seq[j]
                
            result = ''.join(chars)
            results.append(result)
        return results

# Step 2: Load the dataset
print("Loading dataset from huggingface")
ds = load_dataset("phunc20/nj_biergarten_captcha")
print("Dataset loaded")

#def extract_labels(dataset):
#    return [''.join(key.split('_')[1:]).lower() for key in dataset["__key__"]]

alphabet = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

converter = LabelConverter(alphabet)

def extract_labels(dataset):
    # Extract the labels from the dataset
    val = np.array(dataset.split("_")[1:])
    return val

class CustomCaptchaDataset(Dataset):
    #def __init__(self, dataset, transform=None):
    def __init__(self, transform=None, alphabet=None, converter=None):
        self.transform = transform
        self.alphabet = alphabet
        self.converter = converter
        #self.dataset = dataset
        #self.image_files = [np.array(img) for img in dataset["jpg"]]
        #self.labels = extract_labels(dataset)

    def __len__(self):
        return len(ds["train"])

    def __getitem__(self, idx):
        #image = self.image_files[idx]
        image = np.array(ds["train"][idx]["jpg"])
        #print("Image type:", type(image))
        #label = self.labels[idx]
        label = converter.encode(extract_labels(ds["train"][idx]["__key__"]))
        #print("Label:", label)
        #print("Label type:", type(label))

        if self.transform:
            image = self.transform(image)
        return image, label

# Step 3: Image transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Grayscale(),
    transforms.Resize((32, 128)),  # Standard size
    # Add augmentation for better generalization
    transforms.RandomRotation(2),  # Slight rotation
    transforms.RandomAffine(0, translate=(0.05, 0), scale=(0.95, 1.05)),  # Small translations/scaling
    transforms.Normalize((0.5,), (0.5,))  # Standard normalization
])


# Step 4: Prepare data loaders
data_points = 476118
batch_size = 128

print("Creating dataset object...")
dataset = CustomCaptchaDataset(transform=transform, alphabet=alphabet, converter=converter)
train_size = int(data_points * 0.75)
test_size = data_points - train_size
print("Splitting dataset into train and test...")
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
print("Creating data loaders...")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Step 5: Define CRNN model
class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 2)),  # [b, 64, 16, 64]
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 2)),  # [b, 128, 8, 32]
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 2)),  # [b, 256, 4, 16]
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((4, 1)),  # [b, 512, 1, 16]
            nn.Conv2d(512, 512, 1, 1, 0), nn.BatchNorm2d(512), nn.ReLU()  # [b, 512, 1, 16]
        )

        self.rnn1 = nn.LSTM(512, nh, bidirectional=True)
        self.rnn2 = nn.LSTM(nh * 2, nh, bidirectional=True)
        self.embedding = nn.Linear(nh * 2, nclass)

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        assert h == 1, "Height after CNN should be 1"
        conv = conv.squeeze(2)  # [b, c, w]
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        recurrent, _ = self.rnn1(conv)  # [w, b, nh*2]
        recurrent, _ = self.rnn2(recurrent)  # [w, b, nh*2]

        output = self.embedding(recurrent)  # [w, b, nclass]
        return output


# Step 6: Setup model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Creating model...")
model = CRNN(32, 1, len(alphabet) + 1, 256).to(device)
print("Model created")
print("Creating loss function...")
criterion = nn.CTCLoss(blank=0)
print("Loss function created")
print("Creating optimizer...")
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
print("Optimizer created")

# Add this right before your training loop
#print("Checking dataset labels...")
#sample_labels = [train_dataset[i][1] for i in range(min(10, len(train_dataset)))]
#print(f"Sample labels: {sample_labels}")

# Check if all characters exist in the alphabet
#all_chars = set(''.join(sample_labels))
#missing_chars = [c for c in all_chars if c not in alphabet]
#if missing_chars:
#    print(f"WARNING: These characters are in labels but not in alphabet: {missing_chars}")

# Step 7: Training loop
# Training loop with progress bar and cleaner output
epoch_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

writer = SummaryWriter()
total_batches = 0
best_val_loss = float('inf')

print("Starting training...")
epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    total_train = 0
    total_test = 0
    step = 0

    # Add progress bar for training
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
    
    for images, texts in progress_bar:
    #for batch_data in train_loader:
        #images, texts = batch_data
        step += 1

        images = images.to(device)
        #targets, target_lengths = converter.encode(texts)
        targets = texts[0]
        target_lengths = texts[1]
        targets = targets.to(device)
        #targets = targets.to(torch.float32)
        target_lengths = target_lengths.to(device)

        preds = model(images)
        preds_log_softmax = preds.log_softmax(2)
        
        input_lengths = torch.full((preds.size(1),), preds.size(0), dtype=torch.long).to(device)
        
        loss = criterion(preds_log_softmax, targets, input_lengths, target_lengths)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Calculate accuracy during training
        with torch.no_grad():
            decoded_preds = converter.decode(preds_log_softmax, input_lengths)
            #print("Preds size:", len(decoded_preds))
            #print("Preds", decoded_preds)
            new_list = [converter.encode(decoded_preds[i])[0] for i in range(len(decoded_preds))]
            #convert newlist to tensor and add to device
            #print("Preds encoded", new_list)
            #print("len preds encoded", len(converter.encode(decoded_preds)[0]))
            #print("Targets size:", len(targets))
            #print("Targets", targets)
            for pred, true in zip(new_list, targets):
                pred = pred.to(device)
                #print("Pred:", pred)
                #print("True:", true)
                #if pred == true:
                #if (np.all(torch.eq(pred, true))):
                #    correct += 1
                # Check if the predicted and true tensors are equal, beware of the tensors having different shapes
                if pred.shape == true.shape and torch.all(pred == true):
                    correct += 1
            #print("correct:", correct)
            total += len(targets)
            #print("total:", total)
        
        # Update progress bar with current loss
        train_acc = 100 * correct / total if total > 0 else 0
        progress_bar.set_postfix(loss=f"{total_loss/len(progress_bar):.4f}", accuracy=f"{train_acc:.2f}%")

        # Log loss and accuracy to TensorBoard for total batches
        total_batches += 1
        writer.add_scalar('Total batches', total_loss/(step), total_batches)
    
    avg_train_loss = total_loss / len(train_loader)
    train_accuracy = 100 * correct / total if total > 0 else 0
    
    writer.add_scalar('Train Loss', avg_train_loss, epoch)
    writer.add_scalar('Train Accuracy', train_accuracy, epoch)

    epoch_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)
    
    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    # Add progress bar for validation
    progress_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]")
    
    with torch.no_grad():
        for images, texts in progress_bar:
            images = images.to(device)
            #targets, target_lengths = converter.encode(texts)
            targets = texts[0]
            target_lengths = texts[1]
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            
            preds = model(images)
            preds_log_softmax = preds.log_softmax(2)
            
            input_lengths = torch.full((preds.size(1),), preds.size(0), dtype=torch.long).to(device)
            
            loss = criterion(preds_log_softmax, targets, input_lengths, target_lengths)
            val_loss += loss.item()
            
            decoded_preds = converter.decode(preds_log_softmax, input_lengths)
            new_list = [converter.encode(decoded_preds[i])[0] for i in range(len(decoded_preds))]

            #for pred, true in zip(decoded_preds, texts):
            #    if pred == true:
            #        correct += 1
            for pred, true in zip(new_list, targets):
                pred = pred.to(device)
                print("Pred:", pred)
                print("True:", true)
            
                if pred.shape == true.shape and torch.all(pred == true):
                    correct += 1
            
            print("correct:", correct)
            total += len(targets)
            
            # Update progress bar
            val_acc = 100 * correct / total if total > 0 else 0
            progress_bar.set_postfix(loss=f"{val_loss/len(progress_bar):.4f}", accuracy=f"{val_acc:.2f}%")
   
    avg_val_loss = val_loss / len(test_loader)
    val_accuracy = 100 * correct / total
    
    test_losses.append(avg_val_loss)
    test_accuracies.append(val_accuracy)

    writer.add_scalar('Validation Loss', avg_val_loss, epoch)
    writer.add_scalar('Validation Accuracy', val_accuracy, epoch)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        # Save the model if validation loss improves
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"Model saved at epoch {epoch+1} with validation loss: {avg_val_loss:.4f}")
    
    # Print one simple line with all metrics for this epoch
    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
           


# Plot metrics after training
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), epoch_losses, label='Train Loss')
plt.plot(range(1, epochs+1), test_losses, label='Test Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, epochs+1), test_accuracies, label='Test Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()

# Final test accuracy (calculated during the last validation run)
print(f"Final Test Accuracy: {test_accuracies[-1]:.2f}%")


# # After training, add:
# plt.figure(figsize=(10, 5))
# plt.plot(range(1, epochs+1), epoch_losses)
# plt.title('Training Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.show()

# # After training, add this diagnostic code
# model.eval()
# with torch.no_grad():
#     sample_images, sample_labels = next(iter(test_loader))
#     sample_images = sample_images.to(device)
    
#     # Get CNN feature map dimensions
#     x = sample_images
#     for i, layer in enumerate(model.cnn):
#         x = layer(x)
#         if i % 3 == 2:  # After each pooling layer
#             print(f"After layer {i}: shape = {x.shape}")
            
#     # Final CNN output
#     print(f"Final CNN output shape: {x.shape}")
    
#     # Check sequence length from CNN output
#     print(f"Sequence length from CNN: {x.shape[3]}")
#     print(f"Number of characters to predict: {len(sample_labels[0])}")
    
#     if x.shape[3] < len(sample_labels[0]):
#         print("WARNING: CNN output sequence length is shorter than number of characters!")
#         print("This will likely cause incomplete predictions.")

# # Step 8: Simple evaluation
# model.eval()
# with torch.no_grad():
#     images, labels = next(iter(test_loader))
#     images = images.to(device)
#     preds = model(images)
#     preds_log_softmax = preds.log_softmax(2)
#     pred_lengths = torch.full((images.size(0),), preds.size(0), dtype=torch.long).to(device)

#     decoded_preds = converter.decode(preds_log_softmax, pred_lengths)

#     # Print raw prediction values for the first example
#     print(f"Raw prediction for first example (first few timesteps):")
#     for t in range(min(5, preds.size(0))):
#         print(f"  Timestep {t}: {preds[t, 0, :5]} ... {preds[t, 0, -5:]}")
    
#     # Check if predictions are skewed toward certain indices
#     print(f"Most predicted class: {preds.argmax(2).flatten().bincount().argmax().item()}")

#     for i in range(min(5, len(decoded_preds))):
#         plt.imshow(images[i].cpu().squeeze(), cmap="gray")
#         plt.title(f"Pred: {decoded_preds[i]}, True: {labels[i]}")
#         plt.axis('off')
#         plt.show()

# # Step 9: Compute accuracy on test set
# correct = 0
# total = 0

# model.eval()
# with torch.no_grad():
#     for images, labels in test_loader:
#         images = images.to(device)
#         preds = model(images)
#         preds_log_softmax = preds.log_softmax(2)
#         pred_lengths = torch.full((images.size(0),), preds.size(0), dtype=torch.long).to(device)

#         decoded_preds = converter.decode(preds_log_softmax, pred_lengths)

#         for pred, true in zip(decoded_preds, labels):
#             if pred == true:
#                 correct += 1
#         total += len(labels)

# accuracy = correct / total * 100
# print(f"Test Accuracy: {accuracy:.2f}%")



