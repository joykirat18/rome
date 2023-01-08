model="EleutherAI/gpt-neo-1.3B"
model_layer_size=49 
# 49 # gpt2-xl
# 25 #"EleutherAI/gpt-neo-1.3B"

python3 causal_trace.py --model_name $model --prompt "8 + 8 =" --subject "8 + 8" --model_layer_size $model_layer_size
python3 causal_trace.py --model_name $model --prompt "One hundred plus two hundred is equal to" --subject "One hundred plus two hundred" --model_layer_size $model_layer_size
python3 causal_trace.py --model_name $model --prompt "Two multiplied by two is equal to" --subject "Two multiplied by two" --model_layer_size $model_layer_size
python3 causal_trace.py --model_name $model --prompt "Question: Tom has 10 oranges. Lucy has 5 more. How many oranges do they have? Answer: Let's break it down into multiple steps. Initially, Tom has var1 oranges. Lucy has var2 oranges So they have (var1+var2) oranges in total. Here, var1 = 10, var2 = 5. So they have (10+5) =" --subject "Tom has 10 oranges. Lucy has 5 more" --model_layer_size $model_layer_size




