model="EleutherAI/gpt-neox-20b" # EleutherAI/gpt-neox-20b
python3 hiddenValue.py --prompt "Seven plus two is equal to" --model $model
python3 hiddenValue.py --prompt "8 + 8 =" --model $model
python3 hiddenValue.py --prompt "One hundred plus two hundred is equal to" --model $model
python3 hiddenValue.py --prompt "Two multiplied by two is equal to" --model $model
python3 hiddenValue.py --prompt "Eight multiplied by two is equal to" --model $model
python3 hiddenValue.py --prompt "One two three four five" --model $model
python3 hiddenValue.py --prompt "Question: Tom has 10 oranges. Lucy has 5 more. How many oranges do they have? Answer: Let's break it down into multiple steps. Initially, Tom has var1 oranges. Lucy has var2 oranges So they have (var1+var2) oranges in total. Here, var1 = 10, var2 = 5. So they have (10+5) =" --model $model



 