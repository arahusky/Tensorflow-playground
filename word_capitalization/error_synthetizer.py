import random

# TODO rewrite ala Milan Straka advice
'''
- data: jedna možnost je použít jak říkáte BBC + překlepy. Na generování
překlepů naneštěstí nic neznám (ani jsem nenašel po chvíli googlení),
takže nezbývá než Vámi navrhovaný postup. Jen bych místo čtyř
pravděpodobností použil jen tři -- představte si, že větu generuji
psaním jednotlivých znaků (mezera je považována také za běžný znak).
Když jsem již napsal w_1..w_N a mám napsat w_N+1, tak:
- s pravděpodobností p_1 znak w_N+1 nepřidám
- s pravděpodobností p_2 vložím znak navíc (ideálně by vkládaný znak
měl záviset na w_N a w_N+1, ale klidně to zatím ignorujte) a pak až
w_N+1
- s pravděpodobností p_3 vložím w_N+1 před w_N
Tak může mít chybová věta jiný počet slov (což je sice komplikace, ale
chceme to tak) a p_[123] jsou "absolutní" pravděpodobnosti typu
překlepu
'''


class Error_Synthetizer:
    '''
    This class creates artificial errors in provided texts.
    '''
    letters = 'abcdefghijklmnopqrstuvwxyz'

    def __init__(self, insertion_prob=0.1, deletion_prob=0.1, transposition_prob=0.1):
        self.insertion_prob = insertion_prob
        self.deletion_prob = deletion_prob
        self.transposition_prob = transposition_prob

    def modify_word(self, word):
        result = ''

        for i in range(len(word)):
            if random.random() < self.insertion_prob:
                print('a')
            if random.random() < self.deletion_prob:

                if len(word) < 2:
                    # len(word) must be 1
                    # no transposition
                    if random.random() < self.insertion_prob:
                        # insert either at begin or end
                        if bool(random.getrandbits(1)):
                            word = random.choice(self.letters) + word
                        else:
                            word = word + random.choice(Error_Synthetizer.letters)

                    if random.random() < self.deletion_prob:
                        delete_at_pos = random.randint(0, len(word) - 1)
                        word = word[:delete_at_pos] + word[delete_at_pos + 1:]

                else:
                    # randomly split word in two parts (each part having at least one character)
                    split_index = random.randint(1, len(word) - 1)
                    splitted = (word[:split_index], word[split_index:])

                    if random.random() < self.transposition_prob:
                        last_in_first = splitted[0][-1]
                        splitted[0][-1] = splitted[1][0]
                        splitted[1][0] = last_in_first

                    if random.random() < self.insertion_prob:
                        # insert random character after first part
                        splitted[0] += random.choice(self.letters)

                    if random.random() < self.deletion_prob:
                        # delete last character in first part
                        splitted[0] = splitted[0][:-1]

    def modify_text(self, text):
        print('a')
