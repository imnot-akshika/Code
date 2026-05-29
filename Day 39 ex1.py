from dataclasses import dataclass, field
from typing import Optional
from datetime import date

@dataclass
class Transaction:
    description: str
    amount: float
    category: str
    date: date = field(default_factory=date.today)
    notes: Optional[str] = None

    def is_expense(self) -> bool:
        return self.amount < 0

    def is_income(self) -> bool:
        return self.amount > 0

    def __str__(self) -> str:
        direction = "expense" if self.is_expense() else "income"
        return f"{self.date} | {self.category} | {self.description} | ${abs(self.amount):.2f} ({direction})"

@dataclass
class Budget:
    name: str
    transactions: list[Transaction] = field(default_factory=list)

    def add(self, transaction: Transaction) -> None:
        self.transactions.append(transaction)

    def total(self) -> float:
        return sum(t.amount for t in self.transactions)

    def by_category(self) -> dict[str, float]:
        result = {}
        for t in self.transactions:
            result[t.category] = result.get(t.category, 0) + t.amount
        return result

    def expenses(self) -> list[Transaction]:
        return [t for t in self.transactions if t.is_expense()]

    def income(self) -> list[Transaction]:
        return [t for t in self.transactions if t.is_income()]

    def __str__(self) -> str:
        return f"{self.name}: ${self.total():.2f}"