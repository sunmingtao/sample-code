def parse_time(input_string):
    # Split the input string by "-"
    tokens = input_string.split("-")

    # Initialize variables to store the first and second tokens
    first_token = None
    second_token = None

    # Check if there are at least two tokens
    if len(tokens) >= 2:
        # Assign the first and second tokens to their respective variables
        first_token = tokens[0]
        second_token = tokens[1]

    return first_token, second_token

def time_str_to_seconds(time_str):
    try:
        if len(time_str) == 6:
            hours = int(time_str[:2])
            minutes = int(time_str[2:4])
            seconds = int(time_str[4:])
            total_seconds = (hours * 3600) + (minutes * 60) + seconds
        elif len(time_str) == 4:
            minutes = int(time_str[:2])
            seconds = int(time_str[2:])
            total_seconds = (minutes * 60) + seconds
        else:
            raise ValueError("Input string should be in 'hhmmss' or 'mmss' format.")

        return total_seconds
    except ValueError as e:
        return str(e)