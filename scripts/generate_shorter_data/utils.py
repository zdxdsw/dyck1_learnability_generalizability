import random

def generate(max_depth, depth, max_length, transition):
    if depth < max_depth:
        p = random.random()
        if max_length >= 2:
            if p < transition[0]: # S -> e
                return ''
            elif p < transition[0] + transition[1]: # S -> (S)
                return '(' + generate(max_depth, depth + 1, max_length-2, transition) + ')'
            else: # S -> SS
                return generate(max_depth, depth, max_length//2, transition) + generate(max_depth, depth, max_length//2, transition)
        elif 0 < max_length < 2:
            if p < transition[0]:
                return ''
            else:
                return '(' + generate(max_depth, depth + 1, max_length-2, transition) + ')'
        else:
            return ''
    elif depth == max_depth:
        p = random.random()
        if max_length >= 2:
            if p < transition[0]:
                return ''
            else:
                return generate(max_depth, depth, max_length//2, transition) + generate(max_depth, depth, max_length//2, transition)
        else:
            return ''

def double_check(s):
    """
    If it's valid, return the maximum nesting depth.
    If it's not valid, return -1.
    """
    max_depth = 0
    current_depth = 0
    
    for ch in s:
        if ch == '(':
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif ch == ')':
            current_depth -= 1
            if current_depth < 0:
                return -1
    if current_depth != 0:
        return -1
    return max_depth


# define seven types of "off-by-one" errors
def insert_open(s):
    pos = random.randint(0, len(s))
    return s[:pos] + '(' + s[pos:]
def insert_close(s):
    pos = random.randint(0, len(s))
    return s[:pos] + ')' + s[pos:]
def delete_open(s):
    open_positions = [i for i, ch in enumerate(s) if ch == '(']
    pos = random.choice(open_positions)
    return s[:pos] + s[pos+1:]
def delete_close(s):
    close_positions = [i for i, ch in enumerate(s) if ch == ')']
    pos = random.choice(close_positions)
    return s[:pos] + s[pos+1:]
def swap(s): # pick a random '(' and a random ')' and swap them
    # this may not result in a negative, so we will double-check later
    candidate = "()"
    open_positions = [i for i, ch in enumerate(s) if ch == '(']
    close_positions = [i for i, ch in enumerate(s) if ch == ')']
    pos_open = random.choice(open_positions)
    pos_close = random.choice(close_positions)
    s_list = list(s)
    s_list[pos_open], s_list[pos_close] = s_list[pos_close], s_list[pos_open]
    candidate = ''.join(s_list)
    return candidate
def displace_last_close(s): # the last char is always '(', move it to a random position
    # this may not result in a negative, so we will double-check later
    candidate = "()"
    pos = random.randint(0, len(s)-1)
    candidate = s[:pos] + ')' + s[pos:-1]
    return candidate
def grafting(d, l, pos_len_to_strings):
    # find two shorter, positive, equal-depth strings s1, s2 such that len(s1)+len(s2)+2 = len(s)
    # then construct a negative as s1 + ')' + s2 + '('
    
    # split length, i.e. if l=len(s), find even numbers a, b such that a+b=l, a<=b
    a = -1
    a = random.randrange(2*d, (l-2)//2 + 1, 2)
    b = (l-2) - a

    # find all positive strings of length a and depth d
    s1 = random.choice(pos_len_to_strings[str(a)])
    s2 = random.choice(pos_len_to_strings[str(b)])
    return s1 + ')' + s2 + '('
