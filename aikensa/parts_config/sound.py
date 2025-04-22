import pygame

pygame.mixer.init()
do_sound = pygame.mixer.Sound("aikensa/sound/do.wav") 
re_sound = pygame.mixer.Sound("aikensa/sound/re.wav")
mi_sound = pygame.mixer.Sound("aikensa/sound/mi.wav")
fa_sound = pygame.mixer.Sound("aikensa/sound/fa.wav")
so_sound = pygame.mixer.Sound("aikensa/sound/sol.wav")
la_sound = pygame.mixer.Sound("aikensa/sound/la.wav")
si_sound = pygame.mixer.Sound("aikensa/sound/si.wav")
alarm_sound = pygame.mixer.Sound("aikensa/sound/mixkit-classic-short-alarm-993.wav")
picking_sound = pygame.mixer.Sound("aikensa/sound/mixkit-kids-cartoon-close-bells-2256.wav")
picking_sound_v2 = pygame.mixer.Sound("aikensa/sound/mixkit-page-forward-single-chime-1107.wav")
keisoku_sound = pygame.mixer.Sound("aikensa/sound/tabanete.wav") 
konpou_sound = pygame.mixer.Sound("aikensa/sound/mixkit-software-interface-back-2575.wav")


def play_do_sound():
    do_sound.play()

def play_re_sound():
    re_sound.play()

def play_mi_sound():
    mi_sound.play()

def play_fa_sound():
    fa_sound.play()

def play_sol_sound():
    so_sound.play()

def play_la_sound():
    la_sound.play()

def play_si_sound():
    si_sound.play()

def play_alarm_sound():
    alarm_sound.play() 

def play_picking_sound():
    picking_sound_v2.play()

def play_keisoku_sound():
    keisoku_sound.play()

def play_konpou_sound():
    konpou_sound.play()