import torch
from torch.functional import F


class QuestionAnsweringNeuralFuzzyLogicLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, start_logits, start_positions, end_logits, end_positions):
        # Create an empty tensor of zeros to later use it for one hot encoding

        start_position_one_hot = torch.zeros(start_logits.size(0), start_logits.size(1))

        end_position_one_hot = torch.zeros(end_logits.size(0), end_logits.size(1))

        # Set the bit for the start and end positions.

        start_position_one_hot[torch.arange(start_logits.size(0)), start_positions] = 1.0

        end_position_one_hot[torch.arange(end_logits.size(0)), end_positions] = 1.0

        start_position_one_hot = start_position_one_hot.to(self.device)

        end_position_one_hot = end_position_one_hot.to(self.device)

        # Get predictions

        start_predictions = torch.sigmoid(start_logits)

        end_predictions = torch.sigmoid(end_logits)

        # Turn predictions into fuzzy interpretations

        start_interpretations = torch.where(start_position_one_hot == 0, 1 - start_predictions, start_predictions)

        end_interpretations = torch.where(end_position_one_hot == 0, 1 - end_predictions, end_predictions)

        # Apply combination functions.

        # First apply the combination function across the labels. Start ∧ End for each token

        tokens_interpretations = start_interpretations * end_interpretations  # Product Logic T-Norm

        # Then compute the combination function across all tokens. token_0 ∧ token_1 ∧ ... ∧ token_i-1 ∧ token_i

        tokens_interpretations = torch.prod(tokens_interpretations, dim=1, keepdim=True)

        # Scale the interpretations because i love looking at values greater than 1

        tokens_interpretations = tokens_interpretations.view((-1)) * 10

        # Well since the truth is always 1.0 in fuzzy logic, we need a verum tensor for ground truth.

        verum_interpretation = torch.ones(tokens_interpretations.size(0)) * 10

        # Standard mean squared error

        loss = F.mse_loss(tokens_interpretations, verum_interpretation.to(self.device))

        return loss
