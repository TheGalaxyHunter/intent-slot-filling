"""Conditional Random Field (CRF) layer for sequence labeling.

Implements a first-order linear-chain CRF on top of neural emission scores.
Used in the slot filling head to model label dependencies (e.g., ensuring
that I-tags only follow their corresponding B-tags).

Reference:
  Lafferty, McCallum, Pereira. "Conditional Random Fields: Probabilistic
  Models for Segmenting and Labeling Sequence Data." ICML 2001.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ConditionalRandomField(nn.Module):
    """Linear-chain CRF for structured sequence prediction.

    Computes the conditional log-likelihood during training and uses the
    Viterbi algorithm for decoding at inference time.

    Args:
        num_tags: Number of distinct tags (slot labels).
        batch_first: Whether the first dimension is the batch dimension.
        include_start_end: Whether to include separate start/end transition scores.
    """

    def __init__(
        self,
        num_tags: int,
        batch_first: bool = True,
        include_start_end: bool = True,
    ) -> None:
        super().__init__()
        if num_tags <= 0:
            raise ValueError(f"num_tags must be positive, got {num_tags}")

        self.num_tags = num_tags
        self.batch_first = batch_first

        # Transition score matrix: transitions[i][j] is the score for
        # transitioning from tag i to tag j.
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))

        if include_start_end:
            self.start_transitions = nn.Parameter(torch.randn(num_tags))
            self.end_transitions = nn.Parameter(torch.randn(num_tags))
        else:
            self.start_transitions = None
            self.end_transitions = None

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize transition parameters with uniform distribution."""
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        if self.start_transitions is not None:
            nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        if self.end_transitions is not None:
            nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """Compute the negative log-likelihood of the tag sequence.

        Args:
            emissions: Emission scores from the encoder, shape (B, T, num_tags).
            tags: Ground-truth tag sequence, shape (B, T).
            mask: Boolean mask, shape (B, T). True for valid positions.

        Returns:
            Scalar tensor: mean negative log-likelihood over the batch.
        """
        if not self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)

        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.bool)

        numerator = self._compute_score(emissions, tags, mask)
        denominator = self._compute_log_partition(emissions, mask)

        # Return mean NLL over the batch
        nll = denominator - numerator
        return nll.mean()

    def _compute_score(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """Compute the score of the given tag sequence (numerator).

        The score is the sum of emission scores at each position plus
        transition scores between consecutive tags.
        """
        batch_size, seq_len, _ = emissions.shape

        # Emission score for the first token
        score = emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)

        if self.start_transitions is not None:
            score += self.start_transitions[tags[:, 0]]

        for t in range(1, seq_len):
            current_mask = mask[:, t].float()

            emit_score = emissions[:, t].gather(1, tags[:, t].unsqueeze(1)).squeeze(1)
            trans_score = self.transitions[tags[:, t - 1], tags[:, t]]

            score += (emit_score + trans_score) * current_mask

        # End transition scores
        if self.end_transitions is not None:
            # Find the last valid position for each sequence
            seq_lengths = mask.long().sum(dim=1) - 1
            last_tags = tags.gather(1, seq_lengths.unsqueeze(1)).squeeze(1)
            score += self.end_transitions[last_tags]

        return score

    def _compute_log_partition(
        self,
        emissions: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """Compute the log partition function (denominator) using forward algorithm.

        Uses the log-sum-exp trick for numerical stability.
        """
        batch_size, seq_len, num_tags = emissions.shape

        # Initialize with emission scores at t=0 + start transitions
        alpha = emissions[:, 0]  # (B, num_tags)
        if self.start_transitions is not None:
            alpha = alpha + self.start_transitions.unsqueeze(0)

        for t in range(1, seq_len):
            # alpha_t[j] = log sum_i exp(alpha_{t-1}[i] + trans[i,j] + emit[j])
            emit_scores = emissions[:, t].unsqueeze(1)  # (B, 1, num_tags)
            trans_scores = self.transitions.unsqueeze(0)  # (1, num_tags, num_tags)
            alpha_expand = alpha.unsqueeze(2)              # (B, num_tags, 1)

            inner = alpha_expand + trans_scores + emit_scores  # (B, num_tags, num_tags)
            new_alpha = torch.logsumexp(inner, dim=1)          # (B, num_tags)

            # Only update positions where mask is True
            current_mask = mask[:, t].unsqueeze(1).float()
            alpha = new_alpha * current_mask + alpha * (1.0 - current_mask)

        # Add end transition scores
        if self.end_transitions is not None:
            alpha = alpha + self.end_transitions.unsqueeze(0)

        return torch.logsumexp(alpha, dim=1)  # (B,)

    def decode(
        self,
        emissions: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> list[list[int]]:
        """Find the most likely tag sequence using the Viterbi algorithm.

        Args:
            emissions: Emission scores, shape (B, T, num_tags).
            mask: Boolean mask, shape (B, T).

        Returns:
            List of tag sequences, one per example in the batch.
        """
        if not self.batch_first:
            emissions = emissions.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)

        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=emissions.device)

        batch_size, seq_len, num_tags = emissions.shape

        # Initialize Viterbi variables
        viterbi = emissions[:, 0]  # (B, num_tags)
        if self.start_transitions is not None:
            viterbi = viterbi + self.start_transitions.unsqueeze(0)

        backpointers: list[torch.Tensor] = []

        for t in range(1, seq_len):
            viterbi_expand = viterbi.unsqueeze(2)        # (B, num_tags, 1)
            trans = self.transitions.unsqueeze(0)         # (1, num_tags, num_tags)

            inner = viterbi_expand + trans                # (B, num_tags, num_tags)
            best_scores, best_tags = inner.max(dim=1)     # (B, num_tags) each

            emit = emissions[:, t]                        # (B, num_tags)
            new_viterbi = best_scores + emit

            current_mask = mask[:, t].unsqueeze(1).float()
            viterbi = new_viterbi * current_mask + viterbi * (1.0 - current_mask)
            backpointers.append(best_tags)

        # Add end transitions
        if self.end_transitions is not None:
            viterbi = viterbi + self.end_transitions.unsqueeze(0)

        # Backtrack to find best paths
        best_last_tags = viterbi.argmax(dim=1)  # (B,)
        best_paths: list[list[int]] = []

        for b in range(batch_size):
            seq_length = mask[b].long().sum().item()
            path = [best_last_tags[b].item()]

            for t in range(len(backpointers) - 1, -1, -1):
                if t + 1 >= seq_length:
                    continue
                path.append(backpointers[t][b, path[-1]].item())

            path.reverse()
            best_paths.append(path[:seq_length])

        return best_paths
