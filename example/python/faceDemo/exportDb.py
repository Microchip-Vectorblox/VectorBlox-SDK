import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dbNpy', default='faceDb.npy')
    parser.add_argument('--dbC', default='faceDb.c')
    args = parser.parse_args()

    db = np.load(args.dbNpy, allow_pickle='TRUE')
    
    rangeMax = np.zeros(len(db))
    rangeMin = np.zeros(len(db))
    nameLen = np.zeros(len(db))
    for di,d in enumerate(db):
        rangeMax[di] = np.max(d['embedding']);
        rangeMin[di] = np.min(d['embedding']);
        nameLen[di] = len(d['name'])
      
    # plt.figure(dpi=100); plt.plot(rangeMax);
    # plt.figure(dpi=100); plt.plot(rangeMin);
    # plt.figure(dpi=100); plt.plot(nameLen);
    
    txt = '#define DB_LENGTH {}\n'.format(len(db))
    txt += '#define EMBEDDING_LENGTH {}\n'.format(len(db[0]['embedding']))
    txt += '\n'
    txt += 'const char *db_nameStr[DB_LENGTH] = {\n'
    for d in db:
        txt += '\"{}\",\n'.format(d['name'])
    txt += '};\n'
    txt += '\n'
    txt += 'const int16_t db_embeddings[DB_LENGTH][EMBEDDING_LENGTH] = {\n'
    for d in db:
        emb16 = np.round(d['embedding']*65536).astype('int16')
        txt += '{'
        for x in emb16:
            txt += '{},'.format(x)
        txt += '},\n'
    txt += '};\n'
    
    txtFile = open(args.dbC,"w")
    txtFile.write(txt)
    txtFile.close()
    print('saved database to '+args.dbC)

if __name__ == "__main__":
    main()

