# Transformer 构成
1. Embedding 层

将自然语言的输入处理为机器可以识别的向量。
    
Embedding层是一个存储固定词典的嵌入向量查找表，也就是说，在输入神经网络之前，我们往往会将自然语言输入通过分词器tokennizer，分词器的作用是将自然要语言切分为Token并转化为一个固定的index

例如我们将词表大小设置为4，那么输入“我喜欢你”，分词器可以将输入转化为：

input:我
output:0

input:喜欢
outPut:1

input:你
output:2

2. a
3. a
4. 