from sklearn import linear_model
import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
import warnings

class MultiPartClassifier:
    def __init__(self, num_parts, embedd_dim, model_type='logistic'):
        """
        Initialize the MultiPartClassifier.
        
        Args:
        - num_parts (int): Number of parts (determines the number of classifiers).
        - embedd_dim (int): Embedding dimension of input features.
        - model_type (str): Type of classifier ('logistic' or 'svm').
        """
        if model_type not in ['logistic', 'svm']:
            raise ValueError("model_type must be either 'logistic' or 'svm'")
        
        self.num_parts = num_parts
        self.embedd_dim = embedd_dim
        self.model_type = model_type
        self.is_logistic = 'logistic' == self.model_type
        self.is_svm = 'svm' == self.model_type
        self.classifiers = [
            linear_model.SGDClassifier(loss='log_loss' if model_type == 'logistic' else 'hinge')
            for _ in range(num_parts)
        ]

    def train(self, X, visibility_scores, Y):
        """
        Train the classifiers.
        
        Args:
        - X (numpy.ndarray): Input array of shape [num_samples, num_parts, embedd_dim].
        - visibility_scores (numpy.ndarray): Visibility scores of shape [num_samples, num_parts].
        - Y labels 1 representing target and 0 distractors features, shape of [num_samples]
        """
        num_samples, num_parts, embedd_dim = X.shape

        # Validate shapes
        if embedd_dim != self.embedd_dim:
            raise ValueError(f"Expected embedd_dim={self.embedd_dim}, but got {embedd_dim}")
        if num_parts != self.num_parts:
            raise ValueError(f"Expected num_parts={self.num_parts}, but got {num_parts}")
        if visibility_scores.shape != (num_samples, num_parts):
            raise ValueError(f"Expected visibility_scores shape {(num_samples, num_parts)}, but got {visibility_scores.shape}")

        # Ensure visibility_scores are boolean
        visibility_scores = visibility_scores.astype(bool)

        for part_idx in range(self.num_parts):

            if np.sum(visibility_scores[:, part_idx]) >=1:
                # Prepare data for the current part's classifier
                X_part = X[:, part_idx, :]  # Shape: [num_samples, embedd_dim]
                y_part = Y

                X_part = X_part[visibility_scores[:, part_idx]]
                y_part = y_part[visibility_scores[:, part_idx]]
                
                # Train the classifier for the current part
                self.classifiers[part_idx].partial_fit(X_part, y_part, classes=np.array([0, 1]))
            else:
                continue

    def predict(self, X, visibility_scores):
            """
            Predict using the classifiers.
            
            Args:
            - X (numpy.ndarray): Input array of shape [batch_size, num_parts, embedd_dim].
            - visibility_scores (numpy.ndarray): Visibility scores of shape [batch_size, num_parts].
            
            Returns:
            - predictions (numpy.ndarray): Averaged predictions of shape [batch_size].
            """
            batch_size, num_parts, embedd_dim = X.shape

            # Validate shapes
            if embedd_dim != self.embedd_dim:
                raise ValueError(f"Expected embedd_dim={self.embedd_dim}, but got {embedd_dim}")
            if num_parts != self.num_parts:
                raise ValueError(f"Expected num_parts={self.num_parts}, but got {num_parts}")
            if visibility_scores.shape != (batch_size, num_parts):
                raise ValueError(f"Expected visibility_scores shape {(batch_size, num_parts)}, but got {visibility_scores.shape}")

            # Ensure visibility_scores are boolean
            visibility_scores = visibility_scores.astype(bool)

            # Initialize an array to store predictions
            batch_predictions = np.zeros(batch_size)
            batch_visibilities = np.zeros(batch_size)

            if np.sum(visibility_scores) == 0:
                return np.zeros(batch_size)

            for part_idx in range(self.num_parts):

                try:
                    check_is_fitted(self.classifiers[part_idx], attributes=["coef_"])
                except NotFittedError:
                    print(f"Classifier for part {part_idx} is not fitted. Skipping predictions for this part.")
                    continue

                # Predict for this part
                X_part = X[:, part_idx, :]  # Shape: [batch_size, embedd_dim]
                
                if self.is_logistic:
                    part_predictions = self.classifiers[part_idx].predict_proba(X_part) # Shape: [batch_size]
                    part_predictions = part_predictions[:, 1] # Probability of being correct
                if self.is_svm:
                    part_predictions = self.classifiers[part_idx].decision_function(X_part)  # Shape: [batch_size]

                # Multiply predictions by visibility scores to zero out non-visible parts
                part_predictions = part_predictions * visibility_scores[:, part_idx]

                # Accumulate predictions
                batch_predictions += part_predictions
                batch_visibilities += visibility_scores[:, part_idx]

            if 0 in batch_visibilities:
                visibility_indices = np.where(batch_visibilities != 0)[0]
                batch_predictions[visibility_indices] /= batch_visibilities[visibility_indices]

            else:
                # Divide by the number of visible parts for each sample
                batch_predictions /= batch_visibilities # Avoid division by zero

            return batch_predictions


class MemoryManager:
    def __init__(self, max_samples, parts, embedding_dim):
        """
        Initializes the memory manager with a limit on the number of stored samples.
        Memory is preallocated based on the specified dimensions.

        Args:
        - max_samples (int): The maximum number of samples to store.
        - parts (int): Number of parts per sample.
        - embedding_dim (int): Dimensionality of the embeddings for each part.
        """
        self.max_samples = max_samples
        self.parts = parts
        self.embedding_dim = embedding_dim

        # Preallocate memory for samples, visibility scores, and labels
        self.samples = np.zeros((max_samples, parts, embedding_dim), dtype=float)
        self.visibility_scores = np.zeros((max_samples, parts), dtype=bool)
        self.labels = np.full(max_samples, -1, dtype=int)  # -1 indicates unused slots

        # Track the current number of valid samples
        self.current_size = 0

    def collect(self, batch, visibility_scores, labels, keep=0):
        """
        Collects and stores samples, their labels, and visibility scores with a FIFO policy.
        Optionally retains a certain number of oldest samples when applying FIFO.

        Args:
        - batch (np.ndarray): A batch of samples of shape [batch_size, parts, embedding_dim].
        - visibility_scores (np.ndarray): An array of shape [batch_size, parts].
        - labels (np.ndarray): A 1D array of shape [batch_size] containing 0s and 1s.
        - keep (int, optional): Number of oldest samples to retain when applying FIFO.
        """
        batch_size = batch.shape[0]

        visibility_scores = visibility_scores.astype(bool)

        # Ensure `keep` does not exceed `max_samples`
        if keep > self.max_samples:
            warnings.warn(f"'keep' exceeds max_samples ({self.max_samples}). Adjusting to max_samples.")
            keep = self.max_samples

        # Calculate the number of free slots
        free_slots = self.max_samples - self.current_size

        if batch_size <= free_slots:
            # Case 1: There's enough space to add the entire batch without overwriting
            start_idx = self.current_size
            end_idx = start_idx + batch_size
            self.samples[start_idx:end_idx] = batch
            self.visibility_scores[start_idx:end_idx] = visibility_scores
            self.labels[start_idx:end_idx] = labels
            self.current_size += batch_size
        else:
            # Case 2: Not enough space; apply FIFO policy
            retained_count = min(keep, self.current_size)
            shift = batch_size - free_slots

            # Retain the oldest `keep` samples
            if retained_count > 0:
                self.samples[:retained_count] = self.samples[:retained_count]
                self.visibility_scores[:retained_count] = self.visibility_scores[:retained_count]
                self.labels[:retained_count] = self.labels[:retained_count]

            # Roll existing data to make space for new samples
            self.samples = np.roll(self.samples, -shift, axis=0)
            self.visibility_scores = np.roll(self.visibility_scores, -shift, axis=0)
            self.labels = np.roll(self.labels, -shift, axis=0)

            # Add new samples at the end of the buffer
            start_idx = max(self.max_samples - batch_size, retained_count)
            end_idx = start_idx + batch_size
            self.samples[start_idx:end_idx] = batch[:self.max_samples - start_idx]
            self.visibility_scores[start_idx:end_idx] = visibility_scores[:self.max_samples - start_idx]
            self.labels[start_idx:end_idx] = labels[:self.max_samples - start_idx]
            self.current_size = self.max_samples

    def get_samples(self, last_n=None):
        """
        Retrieves all stored samples, their labels, and visibility scores,
        or optionally the last `n` samples.

        Args:
        - last_n (int, optional): Number of most recent samples to retrieve.

        Returns:
        - tuple: (samples, visibility_scores, labels) if available, otherwise None.
        """
        if self.current_size == 0:
            warnings.warn("No samples available in memory.")
            return None

        if last_n is None or last_n > self.current_size:
            last_n = self.current_size

        indices = np.arange(self.current_size - last_n, self.current_size)
        return self.samples[indices], self.visibility_scores[indices], self.labels[indices]

    def shuffle_samples(self):
        """
        Shuffles the stored samples, their labels, and visibility scores in unison.
        """
        if self.current_size == 0:
            warnings.warn("No samples available to shuffle.")
            return

        # Generate a permutation of indices
        indices = np.random.permutation(self.current_size)
        self.samples[:self.current_size] = self.samples[indices]
        self.labels[:self.current_size] = self.labels[indices]
        self.visibility_scores[:self.current_size] = self.visibility_scores[indices]

    def get_positive_samples(self, last_n=None):
        """
        Retrieves only positive samples, their labels, and visibility scores,
        or optionally the last `n` positive samples.

        Args:
        - last_n (int, optional): Number of most recent positive samples to retrieve.

        Returns:
        - tuple: (positive_samples, positive_visibility_scores, positive_labels) if available, otherwise None.
        """
        mask = self.labels == 1
        positive_samples = self.samples[mask]
        positive_visibility_scores = self.visibility_scores[mask]
        positive_labels = self.labels[mask]

        if last_n is None or last_n > positive_samples.shape[0]:
            last_n = positive_samples.shape[0]

        return (positive_samples[-last_n:], positive_visibility_scores[-last_n:], positive_labels[-last_n:])

    def get_negative_samples(self, last_n=None):
        """
        Retrieves only negative samples, their labels, and visibility scores,
        or optionally the last `n` negative samples.

        Args:
        - last_n (int, optional): Number of most recent negative samples to retrieve.

        Returns:
        - tuple: (negative_samples, negative_visibility_scores, negative_labels) if available, otherwise None.
        """
        mask = self.labels == 0
        negative_samples = self.samples[mask]
        negative_visibility_scores = self.visibility_scores[mask]
        negative_labels = self.labels[mask]

        if last_n is None or last_n > negative_samples.shape[0]:
            last_n = negative_samples.shape[0]

        return (negative_samples[-last_n:], negative_visibility_scores[-last_n:], negative_labels[-last_n:])

    def total_count(self):
        """
        Returns the counts of positive and negative samples.

        Returns:
        - dict: A dictionary containing the counts of positive and negative samples.
        """
        positive_count = np.sum(self.labels[:self.current_size] == 1)
        negative_count = np.sum(self.labels[:self.current_size] == 0)
        return positive_count + negative_count

    def positive_count(self):
        return np.sum(self.labels[:self.current_size] == 1)

    def negative_count(self):
        return np.sum(self.labels[:self.current_size] == 0)
    
    def reset(self):
        # Preallocate memory for samples, visibility scores, and labels
        self.samples = np.zeros((self.max_samples, self.parts, self.embedding_dim), dtype=float)
        self.visibility_scores = np.zeros((self.max_samples, self.parts), dtype=bool)
        self.labels = np.full(self.max_samples, -1, dtype=int)  # -1 indicates unused slots

        # Track the current number of valid samples
        self.current_size = 0