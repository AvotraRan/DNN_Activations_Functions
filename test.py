

import torch


device = "cuda" if torch.cuda.is_available() else "cpu"
def test(model, data_loader, name_of_ac, use_pytorch):
    """Measures the accuracy of a model on a data set."""
    # Make sure the model is in evaluation mode.
    model.eval()
    correct = 0

    # We do not need to maintain intermediate activations while testing.
    with torch.no_grad():

        # Loop over test data.
        for data, target in data_loader:

            # Forward pass.
            output = model(data.to(device))

            # Get the label corresponding to the highest predicted probability.
            pred = output.argmax(dim=1, keepdim=True)

            # Count number of correct predictions.
            correct += pred.cpu().eq(target.view_as(pred)).sum().item()

    # Print test accuracy.
    percent = 100.0 * correct / len(data_loader.dataset)
    
    print(f"Accuracy from {name_of_ac.upper()} activation function "+ ("using" if use_pytorch else "without" )+ f" pytorch implementation : {correct} / {len(data_loader.dataset)} ({percent:.0f}%)")
    print("\n"*3)