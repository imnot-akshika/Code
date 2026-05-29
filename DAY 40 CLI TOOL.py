class Visualiser:
    def __init__(self, data: list[int]):
        self.data = data
        self.steps: list[str] = []
        self.comparisons: int = 0
        self.swaps: int = 0

    def record_step(self, description: str, current_state: list[int]) -> None:
        #store formatted step string
        step = f"{description}: {current_state}"
        self.steps.append(step)

    def print_report(self) -> None:
        print("Sorting Report:")
        print(f"Total Comparisons: {self.comparisons}")
        print(f"Total Swaps: {self.swaps}")
        print("Steps:")
        for step in self.steps:
            print(step)

def visualise_bubble(data: list[int]) -> Visualiser:
    arr = data.copy()
    v = Visualiser(arr)
    n = len(arr)

    for i in range(n):
        for j in range(0, n  - i - 1):
            v.comparisons += 1
            v.record_step(f"Step {len(v.steps)+1}: Comparing index {j} and {j+1}", arr.copy())
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                v.swaps += 1
                v.record_step(f"Step {len(v.steps)+1}: Swap {arr[j+1]} and {arr[j]}", arr.copy())
        
    return v
    
def visualise_insertion(data: list[int]) -> Visualiser:
    arr = data.copy()
    v = Visualiser(arr)

    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            v.comparisons += 1
            v.record_step(f"Step {len(v.steps)+1}: Comparing index {j} and key {key}", arr.copy())
            arr[j + 1] = arr[j]
            j -= 1
            v.swaps += 1
            v.record_step(f"Step {len(v.steps)+1}: Move {arr[j+1]} to index {j+2}", arr.copy())
        arr[j + 1] = key
        v.swaps += 1
        v.record_step(f"Step {len(v.steps)+1}: Insert key {key} at index {j+1}", arr.copy())
        
    return v
    
def visualise_merge(data: list[int]) -> Visualiser:
    arr = data.copy()
    v = Visualiser(arr)
        
    def merge_sort(arr):
        if len(arr) <= 1:
            return arr
        mid = len(arr) // 2
        left = merge_sort(arr[:mid])
        right = merge_sort(arr[mid:])
        merged = []
        i = j = 0
        while i < len(left) and j < len(right):
            v.comparisons += 1
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1
        merged.extend(left[i:])
        merged.extend(right[j:])
        v.record_step(f"Step {len(v.steps)+1}: Merged {left} and {right}", merged.copy())
        return merged
        
    merge_sort(arr)
    return v
    


def visualise_linear(data: list[int], target: int) -> Visualiser:
    arr = data.copy()
    v = Visualiser(arr)

    for i, item in enumerate(arr):
        v.comparisons += 1
        v.record_step(f"Step {len(v.steps)+1}: Comparing index {i} with target {target}", arr.copy())
        if item == target:
            v.record_step(f"Step {len(v.steps)+1}: Found target {target} at index {i}", arr.copy())
            return v
    v.record_step(f"Step {len(v.steps)+1}: Target {target} not found", arr.copy())
    return v
    
def visualise_binary(data: list[int], target: int) -> Visualiser:
    arr = data.copy()
    v = Visualiser(arr)
    low, high = 0, len(arr) - 1

    while low <= high:
        mid = (low + high) // 2
        v.comparisons += 1
        v.record_step(f"Step {len(v.steps)+1}: Comparing index {mid} with target {target}", arr.copy())
        if arr[mid] == target:
            v.record_step(f"Step {len(v.steps)+1}: Found target {target} at index {mid}", arr.copy())
            return v
        elif arr[mid] < target:
            low = mid + 1
            v.record_step(f"Step {len(v.steps)+1}: Target {target} is greater than index {mid}, moving low to {low}", arr.copy())
        else:
            high = mid - 1
            v.record_step(f"Step {len(v.steps)+1}: Target {target} is less than index {mid}, moving high to {high}", arr.copy())
    v.record_step(f"Step {len(v.steps)+1}: Target {target} not found", arr.copy())
    return v


def menu():
    while True:
        print("\n==== Algorithm Visualiser ====")
        print("1. Bubble Sort")
        print("2. Insertion Sort")
        print("3. Merge Sort")
        print("4. Linear Search")
        print("5. Binary Search")
        print("6. Quit")

        choice = input("Select an option: ")
        if choice == '1':
            data = list(map(int, input("Enter numbers to sort (space separated): ").split()))
            v = visualise_bubble(data)
            v.print_report()

        elif choice == '2':
            data = list(map(int, input("Enter numbers to sort (space separated): ").split()))
            v = visualise_insertion(data)
            v.print_report()
        elif choice == '3':
            data = list(map(int, input("Enter numbers to sort (space separated): ").split()))
            v = visualise_merge(data)
            v.print_report()
        elif choice == '4':
            data = list(map(int, input("Enter numbers (space separated): ").split()))
            target = int(input("Enter target number: "))
            v = visualise_linear(data, target)
            v.print_report()
        elif choice == '5':
            data = list(map(int, input("Enter numbers (space separated): ").split()))
            target = int(input("Enter target number: "))
            v = visualise_binary(data, target)
            v.print_report()
        elif choice == '6':
            print("Exiting...")
            break
        else:
            print("Invalid option. Please try again.")

menu()