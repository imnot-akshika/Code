from contextlib import contextmanager

@contextmanager
def managed_file(filename, mode='r'):
    print(f"Opend file {filename}")
    try:
        f = open(filename, mode)
        yield f
    finally:
        f.close()
        print(f"Closed file {filename}")

@contextmanager
def suppress_errors(*exception_types):
    try:
        yield
    except exception_types as e:
        pass


with managed_file("test.txt", "w") as f:
    f.write("Hello!")

with suppress_errors(ValueError, KeyError):
    int("not a number")
print("still running")

# this should NOT be suppressed
try:
    with suppress_errors(ValueError):
        raise TypeError("not suppressed")
except TypeError:
    print("TypeError propagated correctly")