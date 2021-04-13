#!/usr/bin/env python
# coding: utf-8

# In[9]:


"""Этот алгоритм находит решение за 4 попытки"""

import numpy as np

def game_core_v4(number):
    '''Сначала устанавливаем любое random число, а потом уменьшаем или увеличиваем его в зависимости от того, больше оно или меньше нужного.
       Функция принимает загаданное число и возвращает число попыток'''
    count = 1
    predict = np.random.randint(1,101)
    
    while number != predict:
        count+=1
        if number > predict:
            predict = np.random.randint(predict, number) # Определяем диапазон от предполагаемого до загаднного и ищем в нем
            predict += 1
        elif number < predict:
            predict = np.random.randint(number, predict)
            predict -= 1
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

score_game(game_core_v4)

