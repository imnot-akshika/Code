import time

def benchmark():
    # time appending to end
    lst = []
    start = time.time()
    for i in range(100_000):
        lst.append(i)
    append_time = time.time() - start

    # time inserting at start
    lst2 = []
    start = time.time()
    for i in range(1_000):
        lst2.insert(0, i)
    insert_time = time.time() - start

    print(f"Append 100k items: {append_time:.4f}s")
    print(f"Insert 1k at start: {insert_time:.4f}s")
    print(f"Insert is ~{insert_time/append_time * 100:.0f}x slower per operation")

benchmark()


