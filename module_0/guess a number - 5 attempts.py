#!/usr/bin/env python
# coding: utf-8

# In[22]:


"""Этот алгоритм находит решение за 5 попыток"""

import numpy as np

def game_core_v5(number):
    '''Сначала устанавливаем любое random число, а потом уменьшаем или увеличиваем его в зависимости от того, больше оно или меньше нужного.
       Функция принимает загаданное число и возвращает число попыток'''
    lower_bound = 1
    upper_bound = 101
    count = 1
    predict = np.random.randint(lower_bound,upper_bound)
    while number != predict:
        count+=1
        
        '''Перебираем все возможные варианты границы бинарным поиском при условии, что загаданное число больше или меньше предполагаемого'''
        
        if number > predict:
            lower_bound = predict
            # Каждый проход сокращаем дистанцию до загаданного числа
            predict = range(lower_bound, upper_bound)[len(range(lower_bound, upper_bound))//2]
            if number > predict:
                lower_bound = predict
                predict = range(lower_bound, upper_bound)[len(range(lower_bound, upper_bound))//2]
                if number > predict:
                    lower_bound = predict
                    predict = range(lower_bound, upper_bound)[len(range(lower_bound, upper_bound))//2]

        elif number < predict:
            upper_bound = predict
            predict = range(0, upper_bound)[len(range(0, upper_bound))//2]
            if number < predict:
                upper_bound = predict
                predict = range(0, upper_bound)[len(range(0, upper_bound))//2]
                if number < predict:
                    upper_bound = predict
                    predict = range(0, upper_bound)[len(range(0, upper_bound))//2]
    
    return(count) # выход из цикла, если угадали

def score_game(game_core):
    '''Запускаем игру 1000 раз, чтобы узнать, как быстро игра угадывает число'''
    count_ls = []
    np.random.seed(1)  # фиксируем RANDOM SEED, чтобы ваш эксперимент был воспроизводим!
    random_array = np.random.randint(1,101, size=(1000))
    for number in random_array:
        count_ls.append(game_core(number))
    score = int(np.mean(count_ls))
    print(f"Ваш алгоритм угадывает число в среднем за {score} попыток")
    return(score)

score_game(game_core_v5) # Запускаем алгоритм функции от функции


# In[ ]:




