import pygame

pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("aikensa/sound/mixkit-classic-short-alarm-993.wav")
picking_sound = pygame.mixer.Sound("aikensa/sound/mixkit-kids-cartoon-close-bells-2256.wav")
picking_sound_v2 = pygame.mixer.Sound("aikensa/sound/mixkit-page-forward-single-chime-1107.wav")
keisoku_sound = pygame.mixer.Sound("aikensa/sound/tabanete.wav") 
konpou_sound = pygame.mixer.Sound("aikensa/sound/mixkit-software-interface-back-2575.wav")

def play_alarm_sound():
    alarm_sound.play() 

def play_picking_sound():
    picking_sound_v2.play()

def play_keisoku_sound():
    keisoku_sound.play()

def play_konpou_sound():
    konpou_sound.play()