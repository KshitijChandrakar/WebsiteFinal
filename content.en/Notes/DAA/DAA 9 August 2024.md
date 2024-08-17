---
date: 2024-08-09T10:58:08-04:00
categories: 'Notes'
tags: [semester3, DAA]
title: "DAA August 9"
---
#### Suggested Books:
Thomas H Orman - DAA   
[Algorithm Design Manual](https://drive.google.com/file/d/1GQrgfbGIl9AvNB_ySbPF550bTSX00kQM/view?usp=drive_link)




GCD Code:
```python
def gcd(m,n):
	while n != 0 do:
		r = m mod n
		m = n
		n = r
	return m
```

```python
def gcd(m,n):
	t =  min (m,n)
	while true:
		if m%t = 0:
			if n%t = 0:
				return t
		t = t - 1


```

General Approaches To Algorithm Design
- Divide and Conquer
- Greedy Method
- Dynamic Programming
- Basic Search and Traversal
- Graph Theory
- Linear Programming
- Approximation Algorithm
- NP Problem
HW 9 Aug:
Find the prime factors of a given number
