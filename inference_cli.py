from package import Model
print("Starting…"); m = Model()
while True:
    q = input("\nQuestion> ").strip()
    if not q or q.lower() in {"exit","quit"}: break
    ans = m.predict(q)
    print("\nAnswer:", ans, flush=True)
