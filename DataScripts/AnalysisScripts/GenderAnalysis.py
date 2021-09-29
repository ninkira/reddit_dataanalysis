
class GenderAnalysis:
    def __init__(self):
        print("Gender analysis")
#source: https://dvc.org/blog/a-public-reddit-dataset
    def gender_analysis(self,):
        # Import Data
        aita_data = ImportData().load_datafile("aita_top_2.json")

        # first impression: count whenever a person is mentioned in a post, based on community rating
        ahole_contains_mom_counter = 0
        ahole_contains_dad_counter = 0
        ahole_contains_gf_counter = 0
        ahole_contains_bf_counter = 0
        ahole_contains_ex_counter = 0

        nta_contains_mom_counter = 0
        nta_contains_dad_counter = 0
        nta_contains_gf_counter = 0
        nta_contains_bf_counter = 0
        nta_contains_ex_counter = 0
        for idx, post in enumerate(aita_data):
            # text preprocessing
            content = post["content"]
            content_filtered = str(content).translate(str.maketrans('', '', string.punctuation))
            content_filtered = content_filtered.lower()

            verdict = post['verdict']
            print("verdict", verdict)

            if verdict == 'Asshole':


            #print("post_content", content_filtered)
# re.match(r'(?:^|\W)YTA(?:$|\W)', comment_content_filtered):
            #if re.match(r'(?:^|\W)mom(?:$|\W)', content_filtered):
                if content_filtered.__contains__(' mom ') or content_filtered.__contains__(' mother '):
                    print("Text contains Mom")
                    ahole_contains_mom_counter += 1

                if content_filtered.__contains__(' dad ') or content_filtered.__contains__(' father '):
                    print("Text contains Dad")
                    ahole_contains_dad_counter += 1

                #elif re.match(r'(?:^|\W)dad(?:$|\W)', content_filtered) or re.match(r'(?:^|\W)father(?:$|\W)', content_filtered) or  re.match(r'(?:^|\W)papa(?:$|\W)', content_filtered):
                #elif content_filtered.__contains__(' dad ') or content_filtered.__contains__(' father '):
                if content_filtered.__contains__(' gf ') or content_filtered.__contains__(' girlfriend ') or content_filtered.__contains__(' wife '):
                    print("Text contains girlfriend")
                    ahole_contains_gf_counter += 1
                    print("ahole gf counts", ahole_contains_gf_counter)

                if content_filtered.__contains__(' bf ') or content_filtered.__contains__(
                        ' husband ') or content_filtered.__contains__(' boyfriend '):
                    print("Text contains boyfriend")
                    ahole_contains_bf_counter += 1

                if content_filtered.__contains__(
                        ' ex '):
                    print("Text contains ex-gf")
                    ahole_contains_ex_counter += 1

            if verdict == 'Not the A-hole':

                # print("post_content", content_filtered)
                # re.match(r'(?:^|\W)YTA(?:$|\W)', comment_content_filtered):
                # if re.match(r'(?:^|\W)mom(?:$|\W)', content_filtered):
                if content_filtered.__contains__(' mom ') or content_filtered.__contains__(' mother '):
                    print("Text contains Mom")
                    nta_contains_mom_counter += 1

                if content_filtered.__contains__(' dad ') or content_filtered.__contains__(' father '):
                    print("Text contains Dad")
                    nta_contains_dad_counter += 1

                # elif re.match(r'(?:^|\W)dad(?:$|\W)', content_filtered) or re.match(r'(?:^|\W)father(?:$|\W)', content_filtered) or  re.match(r'(?:^|\W)papa(?:$|\W)', content_filtered):
                # elif content_filtered.__contains__(' dad ') or content_filtered.__contains__(' father '):
                if content_filtered.__contains__(' gf ') or content_filtered.__contains__(
                        ' girlfriend ') or content_filtered.__contains__(' wife '):
                    print("Text contains girlfriend")
                    nta_contains_gf_counter += 1
                    print("ahole gf counts", ahole_contains_gf_counter)

                if content_filtered.__contains__(' bf ') or content_filtered.__contains__(
                        ' husband ') or content_filtered.__contains__(' boyfriend '):
                    print("Text contains boyfriend")
                    nta_contains_bf_counter += 1

                if content_filtered.__contains__(
                    ' ex '):
                    print("Text contains ex-gf")
                    nta_contains_ex_counter += 1


        print("a-hole-counters",ahole_contains_bf_counter, ahole_contains_gf_counter, ahole_contains_dad_counter, ahole_contains_mom_counter, ahole_contains_ex_counter)
        print("nta counters", nta_contains_bf_counter, nta_contains_gf_counter, nta_contains_dad_counter,
              nta_contains_mom_counter, nta_contains_ex_counter)
            #elif re.match(r'(?:^|\W)girlfriend(?:$|\W)', content_filtered) or re.match(r'(?:^|\W)wif(?:$|\W)', content_filtered):


        odds_mom = np.log2(ahole_contains_mom_counter / nta_contains_mom_counter)
        odds_dad = np.log2(np.mean(ahole_contains_dad_counter) / np.mean(nta_contains_dad_counter))
        odds_gf = np.log2(np.mean(ahole_contains_gf_counter) / np.mean(nta_contains_gf_counter))
        odds_bf = np.log2(np.mean(ahole_contains_bf_counter) / np.mean(nta_contains_bf_counter))

        print("odds", odds_mom, odds_dad, odds_gf, odds_bf)



        # estimate oddds - based on https://dvc.org/blog/a-public-reddit-dataset
        dataframe = pd.DataFrame()
        for idx, post in enumerate(aita_data):
            # text preprocessing
            content = post["content"]
            verdict = post['verdict']
            content_filtered = str(content).translate(str.maketrans('', '', string.punctuation))
            content_filtered = content_filtered.lower()
            if verdict == "Asshole":
                dataframe = dataframe.append({
                    'text': content_filtered,
                    'is_asshole': 1
                }, ignore_index=True)
            elif verdict == "Not the A-hole":
                dataframe = dataframe.append({
                    'text': content_filtered,
                    'is_asshole': 0
                }, ignore_index=True)

            else:
                continue

        print("dataframe", dataframe['text'])
        dataframe['contains_mom'] = dataframe['text'].str.contains("mom|mother", case=False)
        dataframe['contains_dad'] = dataframe['text'].str.contains("dad|father", case=False)
        dataframe['contains_gf'] = dataframe['text'].str.contains("wife|girlfriend|gf", case=False)
        dataframe['contains_bf'] = dataframe['text'].str.contains("husband|boyfriend|bf", case=False)
        dataframe['contains_ex'] = dataframe['text'].str.contains("ex|ex-bf|ex-gf|ex-boyfriend|ex-girlfriend", case=False)

        yta = dataframe[dataframe['is_asshole'] == 1]
        nta = dataframe[dataframe['is_asshole'] == 0]

        odds_mom = np.log2(np.mean(yta['contains_mom']) / np.mean(nta['contains_mom']))
        odds_dad = np.log2(np.mean(yta['contains_dad']) / np.mean(nta['contains_dad']))
        odds_gf = np.log2(np.mean(yta['contains_gf']) / np.mean(nta['contains_gf']))
        odds_bf = np.log2(np.mean(yta['contains_bf']) / np.mean(nta['contains_bf']))
        odds_ex = np.log2(np.mean(yta['contains_ex']) / np.mean(nta['contains_ex']))

        who = ["Mom", "Dad", "Wife/Girlfriend", "Husband/Boyfriend", "Ex"]
        odds = [odds_mom, odds_dad, odds_gf, odds_bf, odds_ex]

        odds_df = pd.DataFrame(zip(who, odds), columns=["Who", "LogOdds"])
        odds_df['direction'] = odds_df['LogOdds'] > 0
        print("odds df", odds_df)


