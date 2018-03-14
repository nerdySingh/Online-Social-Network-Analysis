"""
sumarize.py
"""
import pickle

def main():
    neg_tweets= pickle.load(open("final_neg_tweets.txt","rb"))
    pos_tweets = pickle.load(open("final_pos_tweets.txt","rb"))
    tweets = pickle.load(open("tweets.txt","rb"))
    users = pickle.load(open("user_screen_names.txt","rb"))
    cluster_count = pickle.load(open("cluster_length.txt","rb"))
    cluster_order = pickle.load(open("cluster_order.txt","rb"))
    users_unique=[]
    count = 0
    for name in users:
        if name not in users_unique:
            users_unique.append(name)
    for i in range(len(cluster_order)):
        count+=cluster_order[i]
    
    summary=[]
    summary.append("Number of users collected:"+str(len(users_unique))+" \n")
    summary.append("Number of messages collected:"+str(len(tweets))+"\n")
    summary.append("Average number of users per community:"+str(count/cluster_count)+"\n")
    summary.append("Number of instances per class found:"+str(len(neg_tweets))+","+str(len(pos_tweets))+"\n")
    summary.append("One example from each class:\n Negative Tweets:"+str(neg_tweets[0])+"\n"+str(" Positive Tweets:"+pos_tweets[0]))
    #print(summary)

    fh = open("summary.txt","w")
    fh.writelines(summary)
    fh.close()
    '''
    fh = open("summary.txt","w")

    fh.write("Number of users collected:",len(users_unique))
    fh.write("Number of messages collected:",len(tweets))
    fh.write("Number of communities discovered:",cluster_count)
    fh.write("Average number of users per community",count/cluster_count)
    fh.write("Number of instances per class found: %d,%d"%(len(neg_tweets),len(pos_tweets)))
    fh.write("One example from each class:\n","","Negative Tweets:\n",neg_tweets[0],"\n","Positive Tweets \n",pos_tweets[0])

    fh.close()
    '''
if __name__=="__main__":
    main()

