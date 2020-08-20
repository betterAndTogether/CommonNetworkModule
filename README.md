# CommonNetworkModule

### Module 1: MultiHeads-SelfAttention(BERT)
* 调用方式
>from SelfAttention import SelfAttention  
>self_attention = SelfAttention(k=[dimension of input], heads=[Heads Num])  
>attention_representation = self_attention(input)
* 说明
> input维度为[batch_size, max_length, k]
* Ref
> https://zhuanlan.zhihu.com/p/93397071


