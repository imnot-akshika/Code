import numpy as np

class GradeBook:
    SUBJECTS = ['Maths', 'Science', 'English', 'History', 'Art']

    def __init__(self, grades: np.ndarray, student_names: list[str]):
        # grades shape: (n_students, n_subjects)
        # validate shape matches
        assert grades.shape == (len(student_names), len(self.SUBJECTS))
        self.grades = grades.astype(float)
        self.names = np.array(student_names)

    def subject_stats(self) -> dict:
        means = np.mean(self.grades, axis=0)
        stds = np.std(self.grades, axis=0)
        minis = np.min(self.grades, axis=0)
        maxis = np.max(self.grades, axis=0)
        return {
            subj: {"mean": means[i], "std": stds[i], "min": minis[i], "max": maxis[i]}
            for i, subj in enumerate(self.SUBJECTS)
        }

    def student_averages(self) -> np.ndarray:
        return np.mean(self.grades, axis=1)

    def top_students(self, n: int = 5) -> list[tuple[str, float]]:
        avgs = self.student_averages()
        sorted_idx = np.argsort(avgs)[::-1]
        top_idx = sorted_idx[:n]
        return [(self.names[i], round(avgs[i], 2)) for i in top_idx]

    def failing_students(self, threshold: float = 50.0) -> list[str]:
        avgs = self.student_averages()
        masks = avgs < threshold
        return list(self.names[masks])

    def subject_correlation(self) -> np.ndarray:
        return np.corrcoef(self.grades.T)

    def grade_distribution(self) -> dict[str, int]:
        g = self.grades.flatten()
        return {
            "A": int(np.sum(g >= 90)),
            "B": int(np.sum((g >= 80) & (g < 90))),
            "C": int(np.sum((g >= 70) & (g < 80))),
            "D": int(np.sum((g >= 60) & (g < 70))),
            "F": int(np.sum(g < 60))
        }

    def normalise_grades(self) -> np.ndarray:
        col_min = np.min(self.grades, axis=0)
        col_max = np.max(self.grades, axis=0)
        return (self.grades - col_min) / (col_max - col_min) * 100

    def add_student(self, name: str, grades: list[float]) -> None:
        new_row = np.array(grades).reshape(1, -1)
        self.grades = np.vstack([self.grades, new_row])
        self.names = np.append(self.names, name)

    def __str__(self) -> str:
        avgs = self.student_averages()
        return (f"GradeBook | {len(self.names)} students | "
                f"5 subjects | Class avg: {np.mean(avgs):.1f}")
    

#Example Usage
np.random.seed(42)
n_students = 30
grades = np.random.randint(40, 100, size=(n_students, 5)).astype(float)

# introduce some failing grades
grades[5, :] = [35, 42, 38, 45, 40]
grades[12, :] = [48, 51, 44, 49, 46]

names = [f"Student_{i:02d}" for i in range(n_students)]
gb = GradeBook(grades, names)

print(gb)
print("\nSubject stats:")
for subj, stats in gb.subject_stats().items():
    print(f"  {subj}: mean={stats['mean']:.1f}, std={stats['std']:.1f}")

print(f"\nTop 5 students: {gb.top_students(5)}")
print(f"Failing students: {gb.failing_students()}")
print(f"\nGrade distribution: {gb.grade_distribution()}")

print("\nCorrelation matrix:")
print(gb.subject_correlation().round(2))

gb.add_student("New_Student", [85, 90, 78, 88, 92])
print(f"\nAfter adding student: {gb}")