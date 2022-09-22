#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 09:22:20 2021

@author: C33880
"""
import numpy as np
import kalmanFilterRect

class faceTracker:
    def __init__(self):
        self.tracks = []
        self.maxResidual = 50
        self.residualWeights = [.5,.5,.5,.5]
        self.similarityAlphaUp = 0.95
        self.similarityAlphaDown = 0.05
        self.detectThresh = 0.45
        self.maxMisses = 2
        self.attributeAlpha = 0.05
        
    def updateDetection(self, faces):
        res = np.zeros((len(faces),len(self.tracks)))
        for f,face in enumerate(faces):
            for t,track in enumerate(self.tracks):
                y = track['filter'].innovationResidual(face['box'])
                res[f,t] = np.linalg.norm(np.multiply(y.transpose(),self.residualWeights))
        
        # match faces to tracks, selecting closest matches greedily
        for face in faces:
            face['track'] = None
        for track in self.tracks:
            track['match'] = False
            track['framesSinceRec'] += 1
        if res.size>0:
            while np.min(res) < self.maxResidual:
                f,t = np.unravel_index(np.argmin(res, axis=None), res.shape)    # indices of closest match
                self.tracks[t]['res'] = res[f,t]
                res[f,:] = self.maxResidual  # don't use this face again
                res[:,t] = self.maxResidual  # don't use this track again
                faces[f]['track'] = self.tracks[t]
                self.tracks[t]['match'] = True
        
        for face in faces:
            if not face['track']:   # start new track
                track = {'match':True, 'filter':kalmanFilterRect.kalmanFilter(face['box']), 'frames':0, 'misses':0, 'res':0.0, 'similarity':0.0, 'name':None, 'framesSinceRec':100, 'gender':None, 'age':None, 'atrMeas':0}
                self.tracks.append(track)
                face['track'] = track
                face['track']['framesSinceRec']
            else:                   # update existing track
                face['track']['filter'].update(face['box'])
                face['track']['frames'] += 1
        
        for track in self.tracks:
            if track['match']:
                track['misses'] = 0
            else:
                track['misses'] += 1    # count number of consecutive frames for which track is not matched
                if track['misses'] <= self.maxMisses:
                    track['filter'].update([])
        self.tracks[:] = [t for t in self.tracks if t['misses']<=self.maxMisses]   # remove unused tracks
        
        faces.sort(reverse=True, key=lambda face: face['track']['framesSinceRec'])
        return faces
    
    def updateRecognition(self, face):
        track = face['track']
        similarity = face['similarity']
        #alphaDown = 1-pow(1-self.similarityAlphaDown,track['framesSinceRec'])
        if face['name'] != track['name']:
            if similarity>self.detectThresh and similarity>track['similarity']:
                alpha = self.similarityAlphaUp
                track['similarity'] = 0.0
                track['name'] = face['name']
            else:
                alpha = self.similarityAlphaDown
                similarity = 0.0
        else:
            if similarity>track['similarity']:
                alpha = self.similarityAlphaUp
            else:
                alpha = self.similarityAlphaDown
        track['similarity'] = alpha*similarity + (1-alpha)*track['similarity']  # exponential smoothing
        track['framesSinceRec'] = 0

    def updateAttribution(self, face):
        track = face['track']
        alpha = self.attributeAlpha
        if track['atrMeas'] == 0:
            track['gender'] = face['gender']
            track['age'] = face['age']
        elif track['atrMeas'] < 1/alpha:
            # average attributes
            track['gender'] = (face['gender'] + track['atrMeas']*track['gender'])/(track['atrMeas']+1)
            track['age'] = (face['age'] + track['atrMeas']*track['age'])/(track['atrMeas']+1)
        else:
             track['gender'] = alpha*face['gender'] + (1-alpha)*track['gender']
             track['age'] = alpha*face['age'] + (1-alpha)*track['age']
        track['atrMeas'] += 1

