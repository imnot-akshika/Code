import numpy as np

class StatAnalyser:
    def __init__(self, data: np.ndarray):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=float)
        self.data = data

    def describe(self) -> str:
        # return: count, mean, std, min, 25th percentile,
        # median, 75th percentile, max
        count = len(self.data)
        mean = np.mean(self.data)
        std = np.std(self.data)
        minimum = np.min(self.data)
        Q1 = np.percentile(self.data, 25)
        median = np.percentile(self.data, 50)
        Q3 = np.percentile(self.data, 75)
        maximum = np.max(self.data)
        return(
            f"{'Statistic':<25}{'Value'}\n"
            f"{'-'*40}\n"
            f"{'Count':<25}{count}\n"
            f"{'Mean':<25}{mean}\n"
            f"{'Standard Deviation':<25}{std}\n"
            f"{'Minimum':<25}{minimum}\n"
            f"{'25th Percentile (Q1)':<25}{Q1}\n"
            f"{'Median (Q2)':<25}{median}\n"
            f"{'75th Percentile (Q3)':<25}{Q3}\n"
            f"{'Maximum':<25}{maximum}"
        )

    def normalise(self) -> np.ndarray:
        # min-max normalisation — scale to [0, 1]
        # (x - min) / (max - min)
        minimum = np.min(self.data)
        maximum = np.max(self.data)
        return (self.data - minimum) / (maximum - minimum)
        
    def standardise(self) -> np.ndarray:
        # z-score standardisation — mean=0, std=1
        # (x - mean) / std
        mean = np.mean(self.data)
        std = np.std(self.data)
        return (self.data - mean) / std
        
    def outliers(self, threshold: float = 2.0) -> np.ndarray:
        # return values where |z-score| > threshold
        z_scores = self.standardise()
        return self.data[np.abs(z_scores) > threshold]

    def moving_average(self, window: int) -> np.ndarray:
        # compute moving average with given window size
        # hint: np.convolve with np.ones(window)/window
        arr = self.data

        kernel = np.ones(window)/window
        moving_average =  np.convolve(arr, kernel, mode='valid')
        return moving_average

    def correlation(self, other: np.ndarray) -> float:
        # return Pearson correlation coefficient with another array
        # hint: np.corrcoef returns a 2×2 matrix
        matrix = np.corrcoef(self.data, other)[0, 1]

        return matrix
        

    def histogram(self, bins: int = 5) -> tuple[np.ndarray, np.ndarray]:
        # return (counts, bin_edges) using np.histogram
        counts, bin_edges = np.histogram(self.data, bins=bins)
        return counts, bin_edges
    



#Exnp.random.seed(42)
np.random.seed(42)
data = np.random.randn(100) * 15 + 50    # ~normal, mean=50, std=15

analyser = StatAnalyser(data)

print(analyser.describe())
print(f"Normalised range: {analyser.normalise().min():.2f} to {analyser.normalise().max():.2f}")
print(f"Standardised mean: {analyser.standardise().mean():.6f}")
print(f"Outliers: {analyser.outliers(2.0)}")

prices = np.random.randn(100) * 10 + 100
print(f"Correlation with prices: {analyser.correlation(prices):.4f}")

counts, edges = analyser.histogram(bins=5)
for i in range(len(counts)):
    print(f"  {edges[i]:.1f}–{edges[i+1]:.1f}: {counts[i]} values")

ma = analyser.moving_average(10)
print(f"Moving average shape: {ma.shape}")