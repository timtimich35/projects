#!/usr/bin/env python
# coding: utf-8

# In[9]:


"""Этот алгоритм находит решение за 2 попытки"""

import numpy as np

range_list = [] # Определяем список, в который добавим несколько диапазонов
for i in range(0, 101):
    if i > 0 and i % 5 == 0: # Устанавливаем правило для опредления диапазонов 
        range_list.append(range(i-5, i+1)) # Добавляем в наш список 20 диапазонов 0-5, 5-10, 10-15 и т.д. до 95-100
        
def game_core_v3(number):
    '''Сначала устанавливаем любое random число, а потом уменьшаем или увеличиваем его в зависимости от того, больше оно или меньше нужного.
       Функция принимает загаданное число и возвращает число попыток'''
    count = 1
    predict = np.random.randint(1,101) # По сути лишняя операция, т.к. мы переопределяем значение переменной
    
    for item in range_list: # Находим в каком диапазоне лежит загаданное число и берем средний элемент для поиска из этого диапазона 
        if number in item:
            predict = item[2]
            
    while number != predict: # Ищем больше или меньше наше значение искомого и в итоге находим его
        count+=1
        if number > predict: 
            predict += 1
        elif number < predict: 
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

score_game(game_core_v3)

