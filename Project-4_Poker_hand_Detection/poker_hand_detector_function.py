
def poker_hand_detector(hand):
    rank = []
    suit = []
    rank_dict = {'2': 2,
                 '3': 3,
                 '4': 4,
                 '5': 5,
                 '6': 6,
                 '7': 7,
                 '8': 8,
                 '9': 9,
                 '10': 10,
                 'J': 11,
                 'Q': 12,
                 'K': 13,
                 'A': 14
                 }
    possible_hands = []

    poker_hand_dict = {
        1: 'High Card',
        2: 'Pair',
        3: 'Two Pair',
        4: 'Three of a Kind',
        5: 'Straight',
        6: 'Flush',
        7: 'Full House',
        8: 'Four of a Kind',
        9: 'Straight Flush',
        10: 'Royal Flush'
    }
    for card in hand:
        rank.append(rank_dict[card[:-1]])
        suit.append(card[-1])

    sorted_rank = sorted(rank)

    # Straight
    straight = True
    k = sorted_rank[0]
    for i in range(len(sorted_rank)):
        if k != sorted_rank[i]:
            straight = False
            break
        k += 1

    # Flush
    flush = True
    suit_set = list(set(suit))
    rank_set = list(set(rank))
    if len(suit_set) != 1:
        flush = False


    if flush and straight and sorted_rank[0] == 10:                    # Royal Flush
        possible_hands.append(10)
    elif flush and straight:                                           # Straight Flush
        possible_hands.append(9)
    elif len(rank_set) == 2 and (rank.count(rank_set[0]) == 4 or rank.count(rank_set[1]) == 4):      # Four of a Kind
        possible_hands.append(8)
    elif len(rank_set) == 2 and rank.count(rank_set[0]) in [2,3]:      # Full House
        possible_hands.append(7)
    elif flush:                                                        # Flush
        possible_hands.append(6)
    elif straight:                                                     # Straight
        possible_hands.append(5)
    elif len(rank_set) == 3 and (rank.count(rank_set[0]) == 3 or rank.count(rank_set[1]) == 3 or rank.count(rank_set[2]) == 3):      # Three of a Kind
        possible_hands.append(4)
    elif len(rank_set) == 3 and (rank.count(rank_set[0]) == 2 or rank.count(rank_set[1]) == 2 or rank.count(rank_set[2]) == 2):      # Two Pair
        possible_hands.append(3)
    elif len(rank_set) == 4 and (rank.count(rank_set[0]) == 2 or rank.count(rank_set[1]) == 2 or rank.count(rank_set[2]) == 2 or rank.count(rank_set[3]) == 2):      # Pair
        possible_hands.append(2)
    else:
        possible_hands.append(1)

    return poker_hand_dict[max(possible_hands)]

if __name__ == '__main__':
    hand = ['10H', '6D', '7C', '7S', '6H']
    print(poker_hand_detector(hand))