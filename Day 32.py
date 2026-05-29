import heapq
from collections import deque

class TaskManager:
    def __init__(self):
        self._urgent = []
        self._regular = deque()
        self._completed = []
        self._index = 0

    def add_task(self,name,  urgent=False, priority=1):
        if urgent is True:
            heapq.heappush(self._urgent, (-priority, self._index, name))
            self._index += 1
        else:
            self._regular.append(name)

    def complete_next(self):
        if not self._urgent and not self._regular:
            return IndexError("No tasks to complete.")
        if self._urgent:
            _, _, name = heapq.heappop(self._urgent)
        else:
            name = self._regular.popleft()
        self._completed.append(name)
        return name

    def undo_last(self):
        if not self._completed:
            return IndexError("No completed tasks to undo.")
        name = self._completed.pop()
        self._regular.appendleft(name)

    def pending_count(self):
        return len(self._urgent) + len(self._regular)

    def __str__(self):
        return f"TaskManager | Urgent: {len(self._urgent)} | Regular: {len(self._regular)} | Completed: {len(self._completed)}"
    

# Example usage:
tm = TaskManager()

tm.add_task("Write report", urgent=False)
tm.add_task("Fix critical bug", urgent=True, priority=10)
tm.add_task("Reply to emails", urgent=False)
tm.add_task("Deploy hotfix", urgent=True, priority=8)
tm.add_task("Update docs", urgent=False)

print(tm)                        # TaskManager | Urgent: 2 | Regular: 3 | Completed: 0

print(tm.complete_next())        # Fix critical bug
print(tm.complete_next())        # Deploy hotfix
print(tm.complete_next())        # Write report

print(tm)                        # TaskManager | Urgent: 0 | Regular: 2 | Completed: 3

tm.undo_last()
print(tm)                        # TaskManager | Urgent: 0 | Regular: 3 | Completed: 2

print(tm.pending_count())        # 3