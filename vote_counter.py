from collections import defaultdict

def count_votes(vote_map, color_list):
    """
    Computes the votes per candidate for each district

    vote_map: {county: {precinct: {candidate: num_votes}}}
    color_list: {color: [(matched_precinct_name, county_name)]}

    returns color_votes: {color: {candidate: num_votes}}
    """
    color_votes = defaultdict(lambda: defaultdict(int))
    for color, precinct_county_list in color_list.items():
        for matched_precinct_name, county_name in precinct_county_list:
            # print(vote_map[matched_precinct_name, county_name)
            # print(list(vote_map[county_name].keys())[3] == 'cherry log')
            # print(type(matched_precinct_name))
            # print(vote_map[county_name]['cherry log'])
            for candidate, num_votes in vote_map[county_name][matched_precinct_name].items():
                color_votes[color][candidate] += num_votes

    return color_votes

def winning_candidates(color_votes):
    """
    color_votes: {color: {candidate: num_votes}}

    returns colors_win: {color: candidate}
    """
    color_wins = {}
    for color, candidate_vote_map in color_votes.items():
        candidate, votes = max(candidate_vote_map.items(), key=lambda tup: tup[1])
        color_wins[color] = candidate
    
    return color_wins

def count_votes_per_party(vote_map, color_list, candidate_party_map):
    """
    returns {color: {party: num_votes}}
    """
    color_votes = count_votes(vote_map, color_list)
    color_wins = defaultdict(lambda: defaultdict(int))
    for color, candidate_vote_map in color_votes.items():
        for candidate, votes in candidate_vote_map.items():
            cand_party = candidate_party_map[candidate]
            color_wins[color][cand_party] += votes
    
    return color_wins


def winning_parties(color_wins, candidate_party_map):
    return {color: candidate_party_map[candidate] for color, candidate in color_wins.items()}

def get_winning_parties(vote_map, color_list, candidate_party_map):
    color_votes = count_votes(vote_map, color_list)
    color_wins = winning_candidates(color_votes)
    return winning_parties(color_wins, candidate_party_map)

def count_republican_districts(color_winning_parties):
    """
    color_winnning_parties: {(color: int): 'democratic' | 'republican' | 'other'}
    """
    return sum(map(lambda p: p == 'republican', color_winning_parties.values()))