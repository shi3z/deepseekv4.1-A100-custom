#!/usr/bin/env python3
"""Quick capability check through the OpenAI-compatible API: fixed items with checkable answers
(arithmetic / word problems, logic, Japanese comprehension, small coding tasks executed in a sandbox).
usage: eval_quick.py [--port 8000] [--model default] [--tag name] [--max-tokens 2048]
Prints per-category accuracy and writes results/eval_quick_<tag>.jsonl."""
import argparse, json, os, re, subprocess, sys, tempfile, time, urllib.request, urllib.error

ITEMS = [
    # (category, prompt, checker) ; checker: ("num", value) | ("str", expected substring, case-insensitive) | ("code", test source)
    ("math", "A train travels 60 km in 45 minutes. At the same speed, how many km does it travel in 2 hours? Answer with the number only.", ("num", 160)),
    ("math", "If 3x + 7 = 31, what is x? Answer with the number only.", ("num", 8)),
    ("math", "What is 17 * 23? Answer with the number only.", ("num", 391)),
    ("math", "A shop sells pens at 3 for $2. How much do 18 pens cost in dollars? Answer with the number only.", ("num", 12)),
    ("math", "What is the sum of all integers from 1 to 100? Answer with the number only.", ("num", 5050)),
    ("math", "Tom has twice as many apples as Ann. Together they have 36. How many does Tom have? Answer with the number only.", ("num", 24)),
    ("math", "What is 2 to the power of 16? Answer with the number only.", ("num", 65536)),
    ("math", "The average of five numbers is 12. Four of them are 8, 10, 14 and 16. What is the fifth? Answer with the number only.", ("num", 12)),
    ("math", "How many prime numbers are there between 10 and 30 inclusive? Answer with the number only.", ("num", 6)),
    ("math", "A rectangle has perimeter 40 and width 6. What is its area? Answer with the number only.", ("num", 84)),
    ("math", "What is 1234 + 4321 - 999? Answer with the number only.", ("num", 4556)),
    ("math", "If a car uses 8 liters per 100 km, how many liters for 350 km? Answer with the number only.", ("num", 28)),
    ("logic", "Alice is older than Bob. Carol is younger than Bob. Who is the oldest? Answer with the name only.", ("str", "alice")),
    ("logic", "All bloops are razzies. All razzies are lazzies. Is every bloop a lazzy? Answer yes or no.", ("str", "yes")),
    ("logic", "Which word does not belong: apple, banana, carrot, grape? Answer with the word only.", ("str", "carrot")),
    ("logic", "I have 3 boxes. One has only apples, one only oranges, one both. All labels are wrong. I pick one fruit from the box labeled 'both' and it is an apple. What does that box contain? Answer with one word.", ("str", "apple")),
    ("logic", "What comes next in the sequence 2, 6, 12, 20, 30? Answer with the number only.", ("num", 42)),
    ("logic", "If yesterday was two days before Friday, what day is tomorrow? Answer with the day only.", ("str", "friday")),
    ("logic", "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost in cents? Answer with the number only.", ("num", 5)),
    ("logic", "How many times does the letter 'r' appear in the word 'strawberry'? Answer with the number only.", ("num", 3)),
    ("ja", "次の文の主語は何ですか。「昨日、駅前で太郎が花子に本を渡した。」名前だけ答えてください。", ("str", "太郎")),
    ("ja", "「猫の手も借りたい」はどういう状況を表す慣用句ですか。「忙しい」という語を含めて一文で答えてください。", ("str", "忙し")),
    ("ja", "日本で一番高い山は何ですか。名前だけ答えてください。", ("str", "富士山")),
    ("ja", "1 ドル 150 円のとき、3,000 円は何ドルですか。数字だけ答えてください。", ("num", 20)),
    ("ja", "「走る」の過去形を答えてください。単語だけ答えてください。", ("str", "走った")),
    ("ja", "次の数列の次の数は？ 1, 1, 2, 3, 5, 8。数字だけ答えてください。", ("num", 13)),
    ("ja", "「東京、大阪、名古屋、札幌」のうち、北海道にある都市はどれですか。都市名だけ答えてください。", ("str", "札幌")),
    ("ja", "500 円の商品を 20% 引きで買うと何円ですか。数字だけ答えてください。", ("num", 400)),
    ("code", "Write a Python function `is_palindrome(s)` that returns True if the string s reads the same forwards and backwards ignoring case and non-alphanumeric characters. Reply with only the code in a ```python block.",
        ("code", "assert is_palindrome('A man, a plan, a canal: Panama')\nassert not is_palindrome('hello')\nassert is_palindrome('')\n")),
    ("code", "Write a Python function `fib(n)` returning the n-th Fibonacci number with fib(0)=0, fib(1)=1, efficient for n=90. Reply with only the code in a ```python block.",
        ("code", "assert fib(0)==0 and fib(1)==1 and fib(10)==55 and fib(90)==2880067194370816120\n")),
    ("code", "Write a Python function `merge_intervals(iv)` that merges overlapping [start, end] intervals given as a list of lists and returns them sorted. Reply with only the code in a ```python block.",
        ("code", "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]])==[[1,6],[8,10],[15,18]]\nassert merge_intervals([[1,4],[4,5]])==[[1,5]]\n")),
    ("code", "Write a Python function `word_freq(text)` returning a dict of lowercase word -> count, words split on whitespace with punctuation .,!? stripped. Reply with only the code in a ```python block.",
        ("code", "assert word_freq('The cat. The dog!')=={'the':2,'cat':1,'dog':1}\n")),
    ("code", "Write a Python function `roman(n)` converting an integer 1..3999 to a Roman numeral string. Reply with only the code in a ```python block.",
        ("code", "assert roman(1994)=='MCMXCIV' and roman(4)=='IV' and roman(3999)=='MMMCMXCIX'\n")),
    ("code", "Write a Python function `two_sum(nums, target)` returning indices [i, j] (i<j) of two numbers adding to target, or None. Reply with only the code in a ```python block.",
        ("code", "assert two_sum([2,7,11,15],9)==[0,1]\nassert two_sum([3,2,4],6)==[1,2]\nassert two_sum([1,2],5) is None\n")),
    ("code", "Write a Python function `balanced(s)` returning True if the brackets ()[]{} in s are balanced. Reply with only the code in a ```python block.",
        ("code", "assert balanced('([]{})') and not balanced('([)]') and balanced('') and not balanced('(')\n")),
    ("code", "Write a Python function `second_largest(xs)` returning the second largest distinct value in a list, or None if it does not exist. Reply with only the code in a ```python block.",
        ("code", "assert second_largest([3,1,4,4,2])==3 and second_largest([5])is None and second_largest([2,2])is None\n")),
    # harder tier
    ("hard-math", "Find the smallest positive integer n such that n is divisible by 6, n+1 is divisible by 7, and n+2 is divisible by 11. Answer with the number only.", ("num", 174)),
    ("hard-math", "How many 5-digit positive integers have digits that are strictly increasing from left to right? Answer with the number only.", ("num", 126)),
    ("hard-math", "A bag has 4 red and 6 blue balls. Two are drawn without replacement. What is the probability both are red, as a reduced fraction a/b? Answer with the fraction only.", ("str", "2/15")),
    ("hard-math", "What is the remainder when 2^100 is divided by 7? Answer with the number only.", ("num", 2)),
    ("hard-math", "The sum of three consecutive odd integers is 141. What is the product of the smallest and largest? Answer with the number only.", ("num", 2205)),
    ("hard-math", "A clock shows 3:40. What is the smaller angle in degrees between the hour and minute hands? Answer with the number only.", ("num", 130)),
    ("hard-math", "How many trailing zeros does 50! have? Answer with the number only.", ("num", 12)),
    ("hard-math", "In how many ways can 8 people be seated around a round table if rotations are considered the same? Answer with the number only.", ("num", 5040)),
    ("hard-logic", "Five houses in a row are painted red, blue, green, yellow, white in some order. Red is immediately left of blue. Green is at one end. Yellow is not next to green. White is in the middle. What color is the first house? Answer with the color only.", ("str", "green")),
    ("hard-logic", "Three people A, B, C: one always lies, two always tell the truth. A says 'B is the liar'. B says 'C is the liar'. C says 'I am not the liar'. Who is the liar? Answer with the letter only.", ("str", "b")),
    ("hard-logic", "You have a 3-liter jug and a 5-liter jug. What is the minimum number of pouring steps (fill, empty, or pour) to measure exactly 4 liters in the 5-liter jug? Answer with the number only.", ("num", 6)),
    ("hard-logic", "A snail climbs 3 m up a 10 m wall each day and slips back 2 m each night. On which day does it reach the top? Answer with the number only.", ("num", 8)),
    ("hard-ja", "太郎は花子より 3 歳年上で、二人の年齢の和は 29 です。5 年後の花子の年齢は？ 数字だけ答えてください。", ("num", 18)),
    ("hard-ja", "「AはBの東、CはAの北、DはCの西」のとき、DはBから見てどの方角にありますか。「北東」のように 2 文字で答えてください。", ("str", "北")),
    ("hard-ja", "ある商品を定価の 2 割引で売ると 1,200 円の利益、3 割引で売ると 300 円の利益が出ます。定価は何円ですか。数字だけ答えてください。", ("num", 9000)),
    ("hard-ja", "次の文の誤りを直した正しい文を一つだけ書いてください: 「彼は毎朝ジョギングをするのが好きだったので、健康的です。」 修正後の文には「健康的」という語を含めてください。", ("str", "健康的")),
    ("hard-code", "Write a Python class `LRUCache(capacity)` with methods get(key) -> value or -1 and put(key, value), both O(1), evicting the least recently used entry. Reply with only the code in a ```python block.",
        ("code", "c=LRUCache(2); c.put(1,1); c.put(2,2); assert c.get(1)==1; c.put(3,3); assert c.get(2)==-1; c.put(4,4); assert c.get(1)==-1 and c.get(3)==3 and c.get(4)==4\n")),
    ("hard-code", "Write a Python function `topo_sort(n, edges)` returning a topological order of nodes 0..n-1 for directed edges [u, v] (u before v), or None if there is a cycle. Reply with only the code in a ```python block.",
        ("code", "o=topo_sort(4,[[0,1],[1,2],[0,3],[3,2]]); assert o.index(0)<o.index(1)<o.index(2) and o.index(3)<o.index(2); assert topo_sort(2,[[0,1],[1,0]]) is None\n")),
    ("hard-code", "Write a Python function `evaluate(expr)` that evaluates an arithmetic expression string with integers, + - * / (integer division truncating toward zero), parentheses and spaces, without using eval. Reply with only the code in a ```python block.",
        ("code", "assert evaluate('2+3*4')==14 and evaluate('(2+3)*4')==20 and evaluate('14-3/2')==13 and evaluate(' 7 * (2 - 5) ')==-21\n")),
    ("hard-code", "Write a Python function `longest_palindrome(s)` returning the longest palindromic substring of s (any one if several). Reply with only the code in a ```python block.",
        ("code", "r=longest_palindrome('babad'); assert r in ('bab','aba'); assert longest_palindrome('cbbd')=='bb'; assert longest_palindrome('a')=='a'\n")),
    ("hard-code", "Write a Python function `dijkstra(n, edges, src)` returning a list of shortest distances from src to every node 0..n-1 for weighted directed edges [u, v, w], with float('inf') for unreachable nodes. Reply with only the code in a ```python block.",
        ("code", "d=dijkstra(4,[[0,1,1],[1,2,2],[0,2,5],[2,3,1]],0); assert d==[0,1,3,4]; assert dijkstra(2,[],0)[1]==float('inf')\n")),
    ("hard-code", "Write a Python function `parse_csv_line(line)` that splits one CSV line into fields, handling double-quoted fields that may contain commas and doubled quotes (\"\") as an escaped quote. Reply with only the code in a ```python block.",
        ("code", "assert parse_csv_line('a,b,c')==['a','b','c']; assert parse_csv_line('\"x,y\",z')==['x,y','z']; assert parse_csv_line('\"say \"\"hi\"\"\",1')==['say \"hi\"','1']; assert parse_csv_line('')==['']\n")),
]

THINK = False

def ask(port, model, prompt, max_tokens, key):
    req = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0}
    if THINK:
        req["thinking"] = True
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(req).encode(), headers=headers), timeout=1800)
    j = json.loads(r.read())
    m = j["choices"][0]["message"]
    return (m.get("content") or ""), (m.get("reasoning_content") or ""), j.get("usage", {}), j.get("timings", {})

def final_answer(text):
    t = text.strip()
    # drop a trailing period, take the last line as the answer candidate
    lines = [l.strip() for l in t.splitlines() if l.strip()]
    return (lines[-1] if lines else t).rstrip('.。')

def check(checker, content):
    kind, exp = checker
    if kind == "num":
        nums = re.findall(r'-?\d[\d,]*\.?\d*', content.replace('，', ','))
        if not nums:
            return False
        cands = [float(n.replace(',', '')) for n in nums]
        return any(abs(c - exp) < 1e-6 for c in cands[-3:])
    if kind == "str":
        return exp.lower() in content.lower()
    if kind == "code":
        m = re.search(r"```(?:python)?\s*(.*?)```", content, re.S)
        src = (m.group(1) if m else content) + "\n" + exp
        with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
            f.write(src); path = f.name
        try:
            p = subprocess.run([sys.executable, path], capture_output=True, timeout=20)
            return p.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        finally:
            os.unlink(path)
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000); ap.add_argument("--model", default="default")
    ap.add_argument("--tag", default="run"); ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--think", action="store_true", help="enable the model's thinking mode")
    a = ap.parse_args()
    global THINK
    THINK = a.think
    key = os.environ.get("DSV41_API_KEY")  # the local server needs none
    os.makedirs("results", exist_ok=True)
    out = open(f"results/eval_quick_{a.tag}.jsonl", "w")
    by_cat = {}
    t_all = time.time()
    for i, (cat, prompt, checker) in enumerate(ITEMS):
        t0 = time.time()
        try:
            content, reasoning, usage, timings = ask(a.port, a.model, prompt, a.max_tokens, key)
            ok = check(checker, content)
            err = None
        except Exception as e:
            content, reasoning, usage, timings, ok, err = "", "", {}, {}, False, str(e)
        dt = time.time() - t0
        by_cat.setdefault(cat, [0, 0]); by_cat[cat][0] += ok; by_cat[cat][1] += 1
        rec = {"i": i, "cat": cat, "ok": ok, "secs": round(dt, 1), "completion_tokens": usage.get("completion_tokens"), "answer": content[-300:], "reasoning_chars": len(reasoning), "error": err}
        out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
        print(f"[{i:2d}] {cat:5s} {'OK ' if ok else 'NG '} {dt:5.1f}s tok={usage.get('completion_tokens')} | {final_answer(content)[:70]!r}", flush=True)
    tot_ok = sum(v[0] for v in by_cat.values()); tot = sum(v[1] for v in by_cat.values())
    print(f"=== {a.tag}: {tot_ok}/{tot} correct ({100*tot_ok/tot:.0f}%) in {time.time()-t_all:.0f}s; " + ", ".join(f"{c} {v[0]}/{v[1]}" for c, v in by_cat.items()))

if __name__ == "__main__":
    main()
