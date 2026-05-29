def enough(cap, on, wait):
    if on == cap:
        return 0
    if on+wait <= cap:
        return 0
    elif on + wait > cap:
        return f"{wait-(cap-on)}"
    

print(enough(100, 60, 50))
print(enough(10, 5, 5))
print(enough(20, 20, 10))
print(enough(69, 67, 74))