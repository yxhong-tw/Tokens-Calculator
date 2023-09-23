import json
from transformers import AutoTokenizer


config = {
    'data_path': 'data/data.json'
    # , 'field': [
    #     'source_category'
    #     , 'article_title'
    #     , 'article_title'
    #     , 'article_content'
    #     , 'article_creation_date'
    # ]
    , 'max_length': 8192
    , 'model_name': 'lmsys/vicuna-7b-v1.5'
}


def load_data():
    with open(file=config['data_path'], mode='r', encoding='UTF-8') as file:
        data = json.loads(file.read())
        file.close()

    return data


def main():
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

    data = load_data()
    data_number = len(data)
    token_number = 0
    token_avg_number = 0

    over_length_string_counter = 0
    over_length_token_counter = 0

    # Does NOT contain over length token data.
    part_token_number = 0
    part_token_avg_number = 0

    for index, one_data in enumerate(data):
        string = ''

        # For only 'article_content'
        string = one_data['article_content']
        if len(string) > config['max_length']:
            print(f'(string) Index: {index}, Len: {len(string)}')
            over_length_string_counter += 1

        # # For multi-fields
        # for key, value in one_data.items():
        #     if key not in config['field']:
        #         continue

        #     if string != '':
        #         string += ','

        #     string += str((key, value))

        tokens = tokenizer.encode(string)
        if len(tokens) > config['max_length']:
            print(f'(token) Index: {index}, Len: {len(tokens)}')
            over_length_token_counter += 1
            part_token_number -= len(tokens)

        token_number += len(tokens)
        part_token_number += len(tokens)

    token_avg_number = token_number / data_number
    part_token_avg_number = \
        part_token_number / (data_number - over_length_token_counter)

    with open(file='output.txt', mode='a', encoding='UTF-8') as file:
        file.write(
            'Details of config:\n'
            + str(config)
            + '\n'
        )
        file.write(f'Number of data: {data_number}\n')
        file.write(f'Total number of tokens: {token_number}\n')
        file.write(f'Average number of tokens: {token_avg_number}\n')
        file.write(
            f'Number of over length string: {over_length_string_counter}\n')
        file.write(
            f'Number of over length token: {over_length_token_counter}\n')
        file.write(f'Part number of tokens: {part_token_number}\n')
        file.write(
            f'Part average number of tokens: {part_token_avg_number}\n\n')


if __name__ == '__main__':
    main()
