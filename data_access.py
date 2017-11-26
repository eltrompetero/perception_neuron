# ================================================================================================ # 
# Module for keeping track of processed data that is ready for analysis. Some classes for organizing
# data are included.
# 
# Author: Eddie Lee edl56@cornell.edu
# ================================================================================================ # 

from __future__ import division
from utils import *
import os
import cPickle as pickle

def get_fnames():
    return ['Eddie L Caeli F March',
          'Eddie F Caeli L March',
          'Eddie J Caeli J Tango',
          'Eddie F Caeli L Tango',
          'Eddie L Caeli F Tango',
          'Caeli (L) Vincent (F) March',
          'Caeli (F) Vincent (L) March',
          'Caeli (L) Vincent (F) Tango',
          'Caeli (F) Vincent (L) Tango',
          'Caeli (J) Vincent (J) Tango',
          'Caeli (L) Vincent (F) Hands',
          'Caeli (F) Vincent (L) Hands',
          'Caeli (J) Vincent (J) Hands',
          'Itai (L) Anja (F) March',
          'Itai (F) Anja (L) March',
          'Itai (L) Anja (F) Tango',
          'Itai (F) Anja (L) Tango',
          'Itai (J) Anja (J) Tango',
          'Itai (L) Anja (F) Hands',
          'Itai (F) Anja (L) Hands',
          'Itai (J) Anja (J) Hands',
          'Itai (J) Anja (J) Hands_1',
          'Caeli (J) Eddie (J) Hands Cal',
          'Caeli (L) Eddie (F) Hands',
          'Caeli (F) Eddie (L) Hands',
          'Caeli (J) Eddie (J) Hands',
          'Caeli (J) Eddie (J) Hands Cal After',
          'Caeli (J) Eddie (J) Hands Cal1 Blind',
          'Caeli (L) Eddie (F) Hands Blind',
          'Caeli (F) Eddie (L) Hands Blind',
          'Caeli (J) Eddie (J) Hands Cal2 Blind',
          'Caeli (J) Sam (J) Hands Cal1',
          'Caeli (L) Sam (F) Hands',
          'Caeli (F) Sam (L) Hands',
          'Caeli (J) Sam (J) Hands Cal2',
          'Caeli (J) Sam (J) Hands',
          'Caeli (J) Eddie (J) Fine Cal 1',
          'Caeli (J) Eddie (J) Fine Cal 2',
          'Caeli (J) Eddie (J) Fine',
          'Caeli (J) Eddie (J) Fine Cal 3',
          'Caeli (J) Eddie (J) Fine Cal 4',
          'Caeli (J) Yunus (J) Cal 1',
          'Caeli (J) Yunus (J) Cal 2',
          'Caeli (L) Yunus (F)',
          'Caeli (J) Yunus (J) Cal 3',
          'Caeli (J) Yunus (J) Cal 4',
          'Caeli (F) Yunus (L)',
          'Caeli (J) Yunus (J) Cal 5',
          'Caeli (J) Yunus (J) Cal 6',
          'Caeli (J) Yunus (J)',
          'Caeli (J) Yunus (J) Cal 7',
          ('Caeli (J) Eddie (J) Half Occlusion','20170307'),
          ('Caeli (J) Eddie (J) Full Occlusion','20170307'),
          ('Caeli (J) Eddie (J) Low Light','20170307'),
          ('Caeli (L) Eddie (F) Hands Startup Timer','20170310'),
          ('Caeli (F) Eddie (L) Hands Startup Timer','20170310'),
          ('Caeli (L) Eddie (F) Hands','20170317'),
          ('Caeli (F) Eddie (L) Hands','20170317'),
          ('Caeli (J) Eddie (J) Hands','20170317'),
          ('Caeli (J) Eddie (J) Hands Half Occlusion','20170317'),
          ('Caeli (J) Eddie (J) Hands Full Occlusion','20170317'),
          ('Caeli (J) Eddie (J) Hands Low Light','20170317'),
          ('Caeli (L) Eddie (F)','20170418'),
          ('Caeli (F) Eddie (L)','20170418'),
          ('Caeli (J) Eddie (J)','20170418'),
          ('Caeli (J) Eddie (J) Right Left Eye Closed','20170418'),
          ('Caeli (J) Eddie (J) Left Right Eye Closed','20170418'),
          ('Caeli (J) Eddie (J) Half Occlusion','20170418'),
          ('Henry (L) Winnie (F)','20170420'),
          ('Henry (F) Winnie (L)','20170420'),
          ('Henry (J) Winnie (J)','20170420'),
          ('Henry (J) Winnie (J) 2','20170420'),
          ('Henry (J) Winnie (J) Low Light','20170420'),
          ('Eddie (L) Freya (F)','20170424'),
          ('Eddie (F) Freya (L)','20170424'),
          ('Eddie (J) Freya (J)','20170424'),
          ('Eddie (J) Freya (J) Low Light','20170424')
          ]

def hand_ix(fileix):
    """
    Return the hand that was used for the trial given the trial number. This is returned as the index to
    bodyparts defined as 
    bodyparts  = [['RightHand','LeftHand'],
                  ['LeftHand','RightHand']]

    Params:
    -------
    fileix (int)
    """
    if type(fileix) is int:
        fileix = str(fileix)
    
    return {'18':1,
            '19':1,
            '20':1,
            '21':1,
            '23':0,
            '24':1,
            '32':0,
            '33':1,
            '35':0,
            '43':0,
            '46':1,
            '49':1,
            '51':1,
            '52':1,
            '53':1,
            '56':0,
            '57':1,
            '58':1,
            '59':1,
            '60':0,
            '61':1,
            '62':1,
            '63':0,
            '64':0,
            '65':1,
            '68':0,
            '69':1,
            '70':1,
            '71':0,
            '72':1,
            '73':0,
            '74':1,
            '75':1}.get(fileix,None)

def global_rotation(fileix):
    """
    Return the angle that the individuals should be rotated by such that they are facing each other across the
    x-axis.

    Params:
    -------
    fileix (int)
    """
    if type(fileix) is int:
        fileix = str(fileix)
    
    return {'43':np.pi/2,
            '46':np.pi/2,
            '49':np.pi/2}.get(fileix,0)


def get_dr(fname,date=None):
    """Return directory where files are saved."""
    from os.path import expanduser
    homedr = expanduser('~')
    datadr = 'Dropbox/Documents/Noitom/Axis Neuron/Motion Files'

    if not date is None:
        return {'20170307':'%s/%s/%s_%s'%(homedr,datadr,date,'Caeli_Eddie_Occlusion'),
                '20170310':'%s/%s/%s_%s'%(homedr,datadr,date,'Caeli_Eddie_Startup'),
                '20170317':'%s/%s/%s_%s'%(homedr,datadr,date,'Caeli_Eddie'),
                '20170418':'%s/%s/%s_%s'%(homedr,datadr,date,'Caeli_Eddie'),
                '20170420':'%s/%s/%s_%s'%(homedr,datadr,date,'Henry_Winnie'),
                '20170424':'%s/%s/%s_%s'%(homedr,datadr,date,'Eddie_Freya')}[date]

    if 'Itai' in fname and 'Anja' in fname:
        return expanduser('~')+'/Dropbox/Documents/Noitom/Axis Neuron/Motion Files/20161205_Itai_Anja/'
    elif 'Caeli' in fname and 'Vincent' in fname:
        return expanduser('~')+'/Dropbox/Documents/Noitom/Axis Neuron/Motion Files/20161130_Caeli_Vincent/'
    elif 'Caeli' in fname and 'Eddie' in fname and 'Startup' in fname:
        return (expanduser('~')+'/Dropbox/Documents/Noitom/Axis Neuron/Motion '+
                'Files/20170310_Caeli_Eddie_Startup/')
    elif 'Caeli' in fname and 'Eddie' in fname and ('Occlusion' in fname or 'Low' in fname):
        return ( expanduser('~')+'/Dropbox/Documents/Noitom/Axis Neuron/Motion '+
                 'Files/20170307_Caeli_Eddie_Occlusion/' )
    elif 'Caeli' in fname and 'Eddie' in fname and 'Blind' in fname:
        return expanduser('~')+'/Dropbox/Documents/Noitom/Axis Neuron/Motion Files/20170127_Caeli_Eddie/'
    elif 'Caeli' in fname and 'Eddie' in fname and not 'Fine' in fname:
        return expanduser('~')+'/Dropbox/Documents/Noitom/Axis Neuron/Motion Files/20170124_Caeli_Eddie/'
    elif 'Caeli' in fname and 'Sam' in fname:
        return expanduser('~')+'/Dropbox/Documents/Noitom/Axis Neuron/Motion Files/20170127_Caeli_Sam/'
    elif 'Caeli' in fname and 'Eddie' in fname and 'Fine' in fname:
        return expanduser('~')+'/Dropbox/Documents/Noitom/Axis Neuron/Motion Files/20170202_Caeli_Eddie/'
    elif 'Caeli' in fname and 'Yunus' in fname:
        return expanduser('~')+'/Dropbox/Documents/Noitom/Axis Neuron/Motion Files/20170203_Caeli_Yunus/'
    else:
        raise Exception("Invalid file name.")

def print_files(ix0=0,ix1=None):
    """
    Print list of available file names with their indices to make it easy to load them.
    2017-01-18

    Params:
    -------
    ix0 (int=0)
    ix1 (int=None)
    """
    fnames=get_fnames()
    ix1 = ix1 or len(fnames)
    fnames = fnames[ix0:ix1]
    for i,f in enumerate(fnames):
        print "%d\t%s"%(i+ix0,f)

def subject_settings_v3(index,return_list=True):
    settings = [{'person':'Zimu3',
                  'modelhandedness':['Left','Right','Left','Right'],
                  'rotation':[0,0,0,0],
                  'trials':['avatar0','avatar1','hand0','hand1']},
                {'person':'Darshna3',
                  'modelhandedness':['Left','Right','Left','Right'],
                  'rotation':[0,0,0,0],
                  'trials':['avatar0','avatar1','hand0','hand1']},
                {'person':'Richard3',
                  'modelhandedness':['Right','Left','Right','Left'],
                  'rotation':[0,0,0,0],
                  'trials':['avatar0','avatar1','hand0','hand1']},
                {'person':'Rachel3',
                  'modelhandedness':['Right','Left','Right','Left'],
                  'rotation':[0,0,0,0],
                  'trials':['avatar0','avatar1','hand0','hand1']},
                {'person':'Priyanka3',
                  'modelhandedness':['Left','Right','Left','Right'],
                  'rotation':[0,0,0,0],
                  'trials':['avatar0','avatar1','hand0','hand1']},
                {'person':'Emily3',
                  'modelhandedness':['Left','Right','Left','Right'],
                  'rotation':[0,0,0,0],
                  'trials':['avatar0','avatar1','hand0','hand1']},
                {'person':'Sam3',
                  'modelhandedness':['Right','Left','Right','Left'],
                  'rotation':[0,0,0,0],
                  'trials':['avatar0','avatar1','hand0','hand1']},
                {'person':'Najila3',
                  'modelhandedness':['Left','Right','Left','Right'],
                  'rotation':[0,0,0,0],
                  'trials':['avatar0','avatar1','hand0','hand1']},
                {'person':'Kemper3',
                  'modelhandedness':['Left','Right','Left','Right'],
                  'rotation':[0,0,0,0],
                  'trials':['avatar0','avatar1','hand0','hand1']},
                {'person':'Lauren3',
                  'modelhandedness':['Right','Left','Right','Left'],
                  'rotation':[0,0,0,0],
                  'trials':['avatar0','avatar1','hand0','hand1']}
                ][index]
    dr = (os.path.expanduser('~')+
      '/Dropbox/Documents/Noitom/Axis Neuron/Motion Files/UE4_Experiments/%s'%settings['person'])
    if return_list:
        output = [settings[k] for k in ['person','modelhandedness','rotation']]
        output.append(dr)
        return output
    return settings,dr

def subject_settings_v3_1(index,return_list=True):
    settings = [{'person':'Eddie3_1',
                  'modelhandedness':['Left'],
                  'rotation':[0],
                  'trials':['avatar']}
                ][index]
    dr = (os.path.expanduser('~')+
      '/Dropbox/Documents/Noitom/Axis Neuron/Motion Files/UE4_Experiments/%s'%settings['person'])
    if return_list:
        output = [settings[k] for k in ['person','modelhandedness','rotation']]
        output.append(dr)
        return output
    return settings,dr


# ------------------ #
# Class definitions. #
# ------------------ #
class VRTrial3_1(object):
    def __init__(self,person,modelhandedness,rotation,dr,fname='trial_dictionaries.p'):
        """
        Parameters
        ----------
        person : str
        modelhandedness : list of str
        rotation : list of float
        dr : str

        Members
        -------
        person
        modelhandedness
        rotation
        dr
        subjectTrial (dict)
            Full Axis Neuron trial data labeled by part+'T' part+'V'.
        templateTrial (dict)
            Full MotionBuilder trial data labeled by part+'T' part+'V'.
        timeSplitTrials
        subjectSplitTrials
        templateSplitTrials

        Methods
        -------
        info
        subject_by_window_dur
        subject_by_window_spec
        pickle_trial_dicts
        pickle_phase
        _fetch_windowspec_indices
        """
        self.person = person
        self.modelhandedness = modelhandedness
        self.rotation = rotation
        self.dr = dr

        # Load gpr data points.
        self.gprmodel = pickle.load(open('%s/%s'%(self.dr,'gpr.p'),'rb'))['gprmodel']
        
        try:
            data = pickle.load(open('%s/%s'%(self.dr,fname),'rb'))
        except Exception:
            self.pickle_trial_dicts(1)
            
        data = pickle.load(open('%s/%s'%(self.dr,fname),'rb'))
        self.templateTrial = data['templateTrial']
        self.subjectTrial = data['subjectTrial']
        self.timeSplitTrials = data['timeSplitTrials']
        self.templateSplitTrials = data['templateSplitTrials']
        self.subjectSplitTrials = data['subjectSplitTrials']
        self.windowsByPart = data['windowsByPart']

        self.trialTypes = ['avatar']

    def info(self):
        print "Person %s"%self.person
        print "Trials available:"
        for part in self.trialTypes:
            print "%s\tInvisible\tTotal"%part
            for spec,_ in self.windowsByPart[part]:
                print "\t%1.2f\t\t%1.2f"%(spec[0],spec[1])
    
    def subject_by_window_dur(self,windowDur,part):
        """
        Params:
        -------
        windowDur (list)
            Duration of visible/invisible cycle.
        part (str)
            Body part to return.
            
        Returns:
        --------
        selection (list)
            List of trials that have given window duration. Each tuple in list is a tuple of the 
            ( (invisible,total window), time, extracted velocity data ).
        """
        raise NotImplementedError("Needs to be fixed.")
        ix = []
        i=0
        for spec,_ in self.windowsByPart[part]:
            if np.isclose(windowDur,spec[1]):
                ix.append(i)
            i += 1
        
        selection = []
        for i in ix:
            selection.append(( self.windowsByPart[part][i][0],
                               self.timeSplitTrials[part][i],
                               self.subjectSplitTrials[part][i] ))
        return selection
    
    def template_by_window_dur(self,windowDur,part):
        raise NotImplementedError("Needs to be fixed.")
        ix = []
        i=0
        for spec,_ in self.windowsByPart[part]:
            if np.isclose(windowDur,spec[1]):
                ix.append(i)
            i += 1
        
        selection = []
        for i in ix:
            selection.append(( self.windowsByPart[part][i][0],
                               self.timeSplitTrials[part][i],
                               self.templateSplitTrials[part][i] ))
        return selection
    
    def subject_by_invisible_dur(self,windowDur,part):
        raise NotImplementedError("Needs to be fixed.")
        ix = []
        i=0
        for spec,_ in self.windowsByPart[part]:
            if np.isclose(windowDur,spec[0]):
                ix.append(i)
            i += 1
        
        selection = []
        for i in ix:
            selection.append(( self.windowsByPart[part][i][0],
                               self.timeSplitTrials[part][i],
                               self.subjectSplitTrials[part][i] ))
        return selection

    def subject_by_window_spec(self,windowSpec,trial_type,precision=None):
        """Automatically search through left and right hand trials."""
        selection = []
        for specix,spec in enumerate(windowSpec):
            ix = self._fetch_windowspec_indices([spec],trial_type,precision=precision)
            
            if len(ix)>0:
                selection.append(( self.windowsByPart[trial_type][ix[0]][0],
                                   self.timeSplitTrials[trial_type][ix[0]],
                                   self.subjectSplitTrials[trial_type][ix[0]] ))
            # Iterate also through hand0 or avatar0, which contains the other hand..
            if trial_type.isalpha():
                selection += self.subject_by_window_spec([windowSpec[specix]],
                                                             trial_type+'0',
                                                             precision=precision)
        return selection

    def template_by_window_spec(self,windowSpec,trial_type,precision=None):
        """Automatically search through left and right hand trials."""
        selection = []
        for specix,spec in enumerate(windowSpec):
            ix = self._fetch_windowspec_indices([spec],trial_type,precision=precision)
            
            if len(ix)>0:
                selection.append(( self.windowsByPart[trial_type][ix[0]][0],
                                   self.timeSplitTrials[trial_type][ix[0]],
                                   self.templateSplitTrials[trial_type][ix[0]] ))
            # Iterate also through hand0 or avatar0, which contains the other hand..
            if trial_type.isalpha():
                selection += self.template_by_window_spec([windowSpec[specix]],
                                                             trial_type+'0',
                                                             precision=precision)
        return selection

    def template_by_invisible_dur(self,windowDur,part):
        raise NotImplementedError("Needs to be fixed.")
        ix = []
        i=0
        for spec,_ in self.windowsByPart[part]:
            if np.isclose(windowDur,spec[0]):
                ix.append(i)
            i += 1
        
        selection = []
        for i in ix:
            selection.append(( self.windowsByPart[part][i][0],
                               self.timeSplitTrials[part][i],
                               self.templateSplitTrials[part][i] ))
        if trialType.isalpha():
            return selection + self.template_by_invisible_dur(windowSpec,trialType+'0')
        return selection

    def visibility_by_window_spec(self,windowSpec,trial_type,precision=None):
        selection = []
        for specix,spec in enumerate(windowSpec):
            ix = self._fetch_windowspec_indices([spec],trial_type,precision=precision)
            
            if len(ix)>0:
                selection.append(( self.windowsByPart[trial_type][ix[0]][0],
                                   self.timeSplitTrials[trial_type][ix[0]],
                                   self.templateSplitTrials[trial_type+'visibility'][ix[0]] ))
            # Iterate also through hand0 or avatar0, which contains the other hand..
            if trial_type.isalpha():
                selection += self.visibility_by_window_spec([windowSpec[specix]],
                                                             trial_type+'0',
                                                             precision=precision)
        return selection

    def phase_by_window_dur(self,source,windowDur,trialType):
        """
        Return instantaneous phase from bandpass filtered velocities on trial specificied by window
        duration.

        Params:
        -------
        source (str)
        windowDur (list of floats)
        trialType (str)
            'avatar', 'avatar0', 'hand', 'hand0'
        """
        raise NotImplementedError("Needs to be fixed.")
        ix = []
        i = 0
        for spec,_ in self.windowsByPart[trialType]:
            if np.isclose(windowDur,spec[1]):
                ix.append(i)
            i += 1
        
        selection = []
        for i in ix:
            try:
                if source=='subject' or source=='s':
                    phases = pickle.load(open('%s/subject_phase_%s_%d.p'%(self.dr,trialType,i),
                                         'rb'))['phases']
                elif source=='template' or source=='t':
                    phases = pickle.load(open('%s/template_phase_%s_%d.p'%(self.dr,trialType,i),
                                         'rb'))['phases']

                phases = [np.vstack(p) for p in phases]
                selection.append(( self.windowsByPart[trialType][i][0],phases ))
            except IOError:
                print "Trial %d in trial type %s not found."%(i,trialType)
        return selection

    def phase_by_window_spec(self,source,windowSpec,trial_type):
        """
        Parameters
        ----------
        source : str
        windowSpec : list
        trial_type : str
        """
        selection = []
        for specix,spec in enumerate(windowSpec):
            ix = self._fetch_windowspec_indices([spec],trial_type,precision=precision)
            
            if len(ix)>0:
                selection.append(( self.windowsByPart[trial_type][ix[0]][0],
                                   self.timeSplitTrials[trial_type][ix[0]],
                                   self.templateSplitTrials[trial_type][ix[0]] ))
                try:
                    if source=='subject' or source=='s':
                        data = pickle.load(open('%s/subject_phase_%s_%d.p'%(self.dr,trial_type,ix[0]),'rb'))
                        phases,vs = data['phases'],data['vs']
                    elif source=='template' or source=='t':
                        data = pickle.load(open('%s/template_phase_%s_%d.p'%(self.dr,trial_type,ix[0]),'rb'))
                        phases,vs = data['phases'],data['vs']
                    
                    phases = [np.vstack(p) for p in phases]
                    selection.append(( self.windowsByPart[trial_type][ix[0]][0],phases ))
                except IOError:
                    print "Trial %d in trial type %s not found."%(ix[0],trial_type)

            # Iterate also through hand0 or avatar0, which contains the other hand.
            if trial_type.isalpha():
                selection += self.phase_by_window_spec(source,
                                                        [windowSpec[specix]],
                                                        trial_type+'0',
                                                        precision=precision)
        return selection

    def filtv_by_window_spec(self,source,windowSpec,trialType,search_all=True):
        """
        Returns:
        --------
        list of twoples (windowSpec, filtv) where filtv is a list of 3 arrays corresponding to each dimension
        """
        raise NotImplementedError()
        ix = self._fetch_windowspec_indices(windowSpec,trialType,precision=precision)
        selection = []

        for i in ix:
            if source=='subject' or source=='s':
                data = pickle.load(open('%s/subject_phase_%s_%d.p'%(self.dr,trialType,i),'rb'))
                phases,vs = data['phases'],data['vs']
            elif source=='template' or source=='t':
                data = pickle.load(open('%s/template_phase_%s_%d.p'%(self.dr,trialType,i),'rb'))
                phases,vs = data['phases'],data['vs']
            else:
                raise Exception

            vs = [np.vstack(p) for p in vs]
            selection.append(( self.windowsByPart[trialType][i][0],vs ))

        if trialType.isalpha() and search_all:
            return selection + self.filtv_by_window_spec(source,windowSpec,trialType+'0',False)

        return selection

    def dphase_by_window_dur(self,windowDur,trialType):
        """
        Difference in phase between subject and template motion.
        """
        raise NotImplementedError
        from misc.angle import mod_angle
        
        subjectPhase = self.phase_by_window_dur('s',windowDur,trialType)
        templatePhase = self.phase_by_window_dur('t',windowDur,trialType)
        dphase = []
        
        for i in xrange(len(subjectPhase)):
            dphase.append(( subjectPhase[i][0], 
                            [mod_angle( s-t ) for s,t in zip(subjectPhase[i][1],templatePhase[i][1])] ))
        if trialType.isalpha():
            return dphase + self.dphase_by_window_dur(windowDur,trialType+'0')
        return dphase

    def dphase_by_window_spec(self,windowSpec,trialType):
        """
        Difference in phase between subject and template motion.
        """
        from misc.angle import mod_angle
        
        subjectPhase = self.phase_by_window_spec('s',windowSpec,trialType)
        templatePhase = self.phase_by_window_spec('t',windowSpec,trialType)
        dphase = []
            
        for i in xrange(len(subjectPhase)):
            dphase.append(( subjectPhase[i][0], 
                            [mod_angle( s-t ) for s,t in zip(subjectPhase[i][1],templatePhase[i][1])] ))
        return dphase

    def pickle_trial_dicts(self,disp=False):
        """
        Put data for analysis into easily accessible pickles. Right now, I extract only visibility and hand
        velocities for AN port data and avatar's motionbuilder files.
        
        Parameters
        ----------
        disp : bool,False
        """
        from pipeline import extract_motionbuilder_model3,extract_AN_port
        from utils import match_time
        from ue4 import load_visibility
        import dill as pickle

        # Load AN data.
        df = pickle.load(open('%s/%s'%(self.dr,'quickload_an_port_vr.p'),'rb'))['df']
        windowsByPart = self.window_specs(self.person,self.dr)

        # Sort trials into the hand, arm, and avatar trial dictionaries: subjectTrial, templateTrial,
        # hmdTrials. These contain arrays for time that were interpolated in for regular sampling and
        # functions for velocities.
        subjectTrial,templateTrial,hmdTrials = {},{},{}
        timeSplitTrials,subjectSplitTrials,templateSplitTrials = {},{},{}

        for trialno,part in enumerate(['avatar']):
            if disp:
                print "Processing %s..."%part
            # Select time interval during which the trial happened.
            if part.isalpha():
                visible,invisible = load_visibility(part+'_visibility',self.dr)
            else:
                visible,invisible = load_visibility(part[:-1]+'_visibility_0',self.dr)
            startEnd = [visible[0],visible[-1]]
            
            # Extract template.
            mbV,mbT = extract_motionbuilder_model3(self.modelhandedness[trialno])
            showIx = mbT < (startEnd[1]-startEnd[0]).total_seconds()
            templateTrial[part+'T'] = mbT[showIx]
            templateTrial[part+'V'] = mbV
            
            # Extract subject from port file.
            anT,anX,anV,anA = extract_AN_port( df,self.modelhandedness[trialno],
                                               rotation_angle=self.rotation[trialno] )
            showIx = (anT>startEnd[0]) & (anT<startEnd[1])
            subjectTrial[part+'T'],subjectTrial[part+'V'] = anT[showIx],anV[0][showIx]
            
            # Put trajectories on the same time samples so we can pipeline our regular
            # computation.
            # Since the AN trial starts after the mbTrial...the offset is positive.
            subjectTrial[part+'V'],subjectTrial[part+'T'] = match_time(subjectTrial[part+'V'],
                   subjectTrial[part+'T'],
                   1/30,
                   offset=0,
                   use_univariate=True)
            
            # Separate the different visible trials into separate arrays.
            # Times for when visible/invisible windows start.
            start = np.zeros((len(visible)+len(invisible)),dtype=object)
            start[::2] = visible
            start[1::2] = invisible
            # Units of seconds.
            start = np.array(map(lambda t:t.total_seconds(),np.diff(start)))
            start = np.cumsum(start)
            invisibleStart = start[::2]  # as seconds
            visibleStart = start[1::2]  # as seconds
            
            # When target is invisible, set visibility to 0.
            visibility = np.ones_like(templateTrial[part+'T'])
            for i,j in zip(invisibleStart,visibleStart):
                assert i<j
                visibility[(templateTrial[part+'T']>=i) & (templateTrial[part+'T']<j)] = 0
            if len(visible)<len(invisible):
                visibility[(templateTrial[part+'T']>=invisible[-1])] = 0
            templateTrial[part+'visibility'] = visibility
            
            # Separate single data take into separate trials.
            timeSplitTrials[part],subjectSplitTrials[part],templateSplitTrials[part] = [],[],[]
            templateSplitTrials[part+'visibility'] = []
            for spec,startendt in windowsByPart[part]:
                startendt = ((startendt[0]-startEnd[0]).total_seconds(),
                             (startendt[1]-startEnd[0]).total_seconds())

                # Save time.
                timeix = (templateTrial[part+'T']<=startendt[1])&(templateTrial[part+'T']>=startendt[0])
                t = templateTrial[part+'T'][timeix]
                timeSplitTrials[part].append(t)

                # Save visibility window.
                templateSplitTrials[part+'visibility'].append( visibility[timeix] )
                
                # Save velocities.
                templateSplitTrials[part].append( templateTrial[part+'V'](t) )
                # Subject sometimes has cutoff window so must reindex time.
                timeix = (subjectTrial[part+'T']<=startendt[1])&(subjectTrial[part+'T']>=startendt[0])
                t = subjectTrial[part+'T'][timeix]
                subjectSplitTrials[part].append( subjectTrial[part+'V'](t) )
            
            timeix = (templateTrial[part+'T']<=invisibleStart[0])&(templateTrial[part+'T']>=0)
            templateSplitTrials[part+'visibility'].insert( 0,visibility[timeix] )
        
        pickle.dump({'templateTrial':templateTrial,
                     'subjectTrial':subjectTrial,
                     'timeSplitTrials':timeSplitTrials,
                     'templateSplitTrials':templateSplitTrials,
                     'subjectSplitTrials':subjectSplitTrials,
                     'windowsByPart':windowsByPart},
                    open('%s/trial_dictionaries.p'%self.dr,'wb'),-1)

    def pickle_phase(self,trial_types=['avatar','avatar0','hand','hand0']):
        """
        Calculate bandpass filtered phase and pickle.
        """
        from pipeline import pipeline_phase_calc
        
        for part in trial_types:
            nTrials = len(self.windowsByPart[part])  # number of trials for that part

            # Subject.
            toProcess = []
            trialNumbers = []
            for i in xrange(nTrials):
                # Only run process if we have data points. Some trials are missing data points.
                # NOTE: At some point the min length should made to correspond to the min window
                # size in the windowing function for filtering.
                if len(self.timeSplitTrials[part][i])>501:
                    trialNumbers.append(i)
                    toProcess.append( (self.timeSplitTrials[part][i],
                                       (self.subjectSplitTrials[part][i][:,0],
                                        self.subjectSplitTrials[part][i][:,1],
                                        self.subjectSplitTrials[part][i][:,2])) )
                else:
                    print "Ignoring %s trial no %d with windowspec (%1.1f,%1.1f)."%(part,i,
                        self.windowsByPart[part][i][0][0],self.windowsByPart[part][i][0][1])
            pipeline_phase_calc(trajs=toProcess,dr=self.dr,
                                file_names=['subject_phase_%s_%d'%(part,i)
                                            for i in trialNumbers])
            # Template.
            toProcess = []
            trialNumbers = []
            for i in xrange(nTrials):
                if len(self.timeSplitTrials[part][i])>501:
                    trialNumbers.append(i)
                    toProcess.append( (self.timeSplitTrials[part][i],
                                       (self.templateSplitTrials[part][i][:,0],
                                        self.templateSplitTrials[part][i][:,1],
                                        self.templateSplitTrials[part][i][:,2])) )
            pipeline_phase_calc(trajs=toProcess,dr=self.dr,
                                file_names=['template_phase_%s_%d'%(part,i)
                                            for i in trialNumbers])

    def _fetch_windowspec_indices(self,specs,trial_type,precision=None):
        """
        Given a particular trial type and a window specification, return all the indices within
        that trial type that match the given specification.  Options for adjusting the
        precision for matching windows.

        Parameters
        ----------
        trial_type : str
        spec : list of tuples
        
        Returns
        -------
        ix : list of ints
        """
        ix = []
        trialWindows = np.array([w[0] for w in self.windowsByPart[trial_type]])
        i = 0  # counter

        if precision is None:
            for spec in specs:
                ix_ = (np.array(spec)[None,:]==trialWindows).all(1)
                if ix_.any():
                    ix.append( np.where(ix_)[0][0] )
        elif type(precision) is float:
            for spec in specs:
                specDiffs = np.abs( trialWindows-np.array(spec)[None,:] )
                ix_ = (specDiffs<=precision).all(1)
                if ix_.any():
                    ix.append(np.where(ix_)[0][0])
        elif type(precision) is tuple:
            for spec in specs:
                specDiffs = np.abs( trialWindows-np.array(spec)[None,:] )
                ix_ = (specDiffs[:,0]<=precision[0])&(specDiffs[:,1]<=precision[1])
                if ix_.any():
                    ix.append(np.where(ix_)[0][0])
        else:
            raise NotImplementedError("precision type not supported.")

        return ix

    def window_specs(self,person,dr):
        """
        Get when the different visible/invisible cycles occur in the given experiment. These data are
        obtained from visibility text files output from UE4.
        
        Parameters
        ----------
        person : str
            Will point to the folder that the data is in.
        dr : str

        Returns
        -------
        windowsByPart : dict
            Keys correspond to trial types. Each dict entry is a list of tuples 
            ((type of window),(window start, window end))
            Window type is a tuple (inv_duration,window_duration)
        """
        from ue4 import load_visibility 

        # Load AN subject data.
        df = pickle.load(open('%s/%s'%(dr,'quickload_an_port_vr.p'),'r'))['df']

        windowsByPart = {}
        
        for trialno,part in enumerate(['avatar']):
            if part.isalpha():
                fname = part+'_visibility'
            else:
                fname = part[:-1]+'_visibility_0'

            visible,invisible = load_visibility(fname,dr)

            # Array denoting visible (with 1) and invisible (with 0) times.
            start = np.zeros((len(visible)+len(invisible)),dtype=object)
            start[::2] = visible
            start[1::2] = invisible
            start = np.array(map(lambda t:t.total_seconds(),np.diff(start)))
            start = np.cumsum(start)
            invisibleStart = start[::2]
            visibleStart = start[1::2]

            # Get the duration of the invisible and visible windows in the time series.
            mxLen = min([len(visibleStart),len(invisibleStart)])
            invDur = np.around(visibleStart[:mxLen]-invisibleStart[:mxLen],1)
            visDur = np.around(invisibleStart[1:][:mxLen-1]-visibleStart[:-1][:mxLen-1],1)
            windowDur = invDur[:-1]+visDur  # total duration cycle of visible and invisible
            
            # Load data saved in gpr.p.
            # The first time point is when the file was written which we can throw out. The second pair of
            # times are when the trial counter is updated immediately after the first fully visible trial. The
            # remaining points are the following trials.
            dataDict = pickle.load(open('%s/%s'%(self.dr,'gpr.p'),'rb'))
            trialStartTimes = dataDict['trialStartTimes']
            trialEndTimes = dataDict['trialEndTimes']
            windowSpecs = []
            windowStart,windowEnd = [],[]
            for i in xrange(len(self.gprmodel.fractions)):
                if i==0:
                    windowSpecs.append((0,0))
                    windowStart.append(visible[0])
                    windowEnd.append(trialStartTimes[1])
                else:
                    invDur = (1-self.gprmodel.fractions[i])*self.gprmodel.durations[i]
                    winDur = self.gprmodel.durations[i]
                    windowSpecs.append((invDur,winDur))

                    windowStart.append(trialStartTimes[i+1])
                    windowEnd.append(trialEndTimes[i+1])

            windowsByPart[part] = zip(windowSpecs,zip(windowStart,windowEnd))

        return windowsByPart

# end VRTrial3_1

class VRTrial3(object):
    def __init__(self,person,modelhandedness,rotation,dr,fname='trial_dictionaries.p'):
        """
        Params:
        -------
        person (str)
        modelhandedness (list of str)
        rotation (list of float)
        dr (str)

        Attributes:
        -----------
        person
        modelhandedness
        rotation
        dr
        subjectTrial (dict)
            Full Axis Neuron trial data labeled by part+'T' part+'V'.
        templateTrial (dict)
            Full MotionBuilder trial data labeled by part+'T' part+'V'.
        timeSplitTrials
        subjectSplitTrials
        templateSplitTrials

        Methods:
        --------
        info
        subject_by_window_dur
        subject_by_window_spec
        pickle_trial_dicts
        pickle_phase
        _fetch_windowspec_indices
        """
        self.person = person
        self.modelhandedness = modelhandedness
        self.rotation = rotation
        self.dr = dr
        
        try:
            data = pickle.load(open('%s/%s'%(self.dr,fname),'rb'))
        except Exception:
            self.pickle_trial_dicts(1)
            
        data = pickle.load(open('%s/%s'%(self.dr,fname),'rb'))
        self.templateTrial = data['templateTrial']
        self.subjectTrial = data['subjectTrial']
        self.timeSplitTrials = data['timeSplitTrials']
        self.templateSplitTrials = data['templateSplitTrials']
        self.subjectSplitTrials = data['subjectSplitTrials']
        self.windowsByPart = data['windowsByPart']

    def info(self):
        print "Person %s"%self.person
        print "Trials available:"
        for part in ['avatar','avatar0','hand','hand0']:
            print "%s\tInvisible\tTotal"%part
            for spec,_ in self.windowsByPart[part]:
                print "\t%1.2f\t\t%1.2f"%(spec[0],spec[1])
    
    def subject_by_window_dur(self,windowDur,part):
        """
        Params:
        -------
        windowDur (list)
            Duration of visible/invisible cycle.
        part (str)
            Body part to return.
            
        Returns:
        --------
        selection (list)
            List of trials that have given window duration. Each tuple in list is a tuple of the 
            ( (invisible,total window), time, extracted velocity data ).
        """
        raise NotImplementedError("Needs to be fixed.")
        ix = []
        i=0
        for spec,_ in self.windowsByPart[part]:
            if np.isclose(windowDur,spec[1]):
                ix.append(i)
            i += 1
        
        selection = []
        for i in ix:
            selection.append(( self.windowsByPart[part][i][0],
                               self.timeSplitTrials[part][i],
                               self.subjectSplitTrials[part][i] ))
        return selection
    
    def template_by_window_dur(self,windowDur,part):
        raise NotImplementedError("Needs to be fixed.")
        ix = []
        i=0
        for spec,_ in self.windowsByPart[part]:
            if np.isclose(windowDur,spec[1]):
                ix.append(i)
            i += 1
        
        selection = []
        for i in ix:
            selection.append(( self.windowsByPart[part][i][0],
                               self.timeSplitTrials[part][i],
                               self.templateSplitTrials[part][i] ))
        return selection
    
    def subject_by_invisible_dur(self,windowDur,part):
        raise NotImplementedError("Needs to be fixed.")
        ix = []
        i=0
        for spec,_ in self.windowsByPart[part]:
            if np.isclose(windowDur,spec[0]):
                ix.append(i)
            i += 1
        
        selection = []
        for i in ix:
            selection.append(( self.windowsByPart[part][i][0],
                               self.timeSplitTrials[part][i],
                               self.subjectSplitTrials[part][i] ))
        return selection

    def subject_by_window_spec(self,windowSpec,trial_type,precision=None):
        """Automatically search through left and right hand trials."""
        selection = []
        for specix,spec in enumerate(windowSpec):
            ix = self._fetch_windowspec_indices([spec],trial_type,precision=precision)
            
            if len(ix)>0:
                selection.append(( self.windowsByPart[trial_type][ix[0]][0],
                                   self.timeSplitTrials[trial_type][ix[0]],
                                   self.subjectSplitTrials[trial_type][ix[0]] ))
            # Iterate also through hand0 or avatar0, which contains the other hand..
            if trial_type.isalpha():
                selection += self.subject_by_window_spec([windowSpec[specix]],
                                                             trial_type+'0',
                                                             precision=precision)
        return selection

    def template_by_window_spec(self,windowSpec,trial_type,precision=None):
        """Automatically search through left and right hand trials."""
        selection = []
        for specix,spec in enumerate(windowSpec):
            ix = self._fetch_windowspec_indices([spec],trial_type,precision=precision)
            
            if len(ix)>0:
                selection.append(( self.windowsByPart[trial_type][ix[0]][0],
                                   self.timeSplitTrials[trial_type][ix[0]],
                                   self.templateSplitTrials[trial_type][ix[0]] ))
            # Iterate also through hand0 or avatar0, which contains the other hand..
            if trial_type.isalpha():
                selection += self.template_by_window_spec([windowSpec[specix]],
                                                             trial_type+'0',
                                                             precision=precision)
        return selection

    def template_by_invisible_dur(self,windowDur,part):
        raise NotImplementedError("Needs to be fixed.")
        ix = []
        i=0
        for spec,_ in self.windowsByPart[part]:
            if np.isclose(windowDur,spec[0]):
                ix.append(i)
            i += 1
        
        selection = []
        for i in ix:
            selection.append(( self.windowsByPart[part][i][0],
                               self.timeSplitTrials[part][i],
                               self.templateSplitTrials[part][i] ))
        if trialType.isalpha():
            return selection + self.template_by_invisible_dur(windowSpec,trialType+'0')
        return selection

    def visibility_by_window_spec(self,windowSpec,trial_type,precision=None):
        selection = []
        for specix,spec in enumerate(windowSpec):
            ix = self._fetch_windowspec_indices([spec],trial_type,precision=precision)
            
            if len(ix)>0:
                selection.append(( self.windowsByPart[trial_type][ix[0]][0],
                                   self.timeSplitTrials[trial_type][ix[0]],
                                   self.templateSplitTrials[trial_type+'visibility'][ix[0]] ))
            # Iterate also through hand0 or avatar0, which contains the other hand..
            if trial_type.isalpha():
                selection += self.visibility_by_window_spec([windowSpec[specix]],
                                                             trial_type+'0',
                                                             precision=precision)
        return selection

    def phase_by_window_dur(self,source,windowDur,trialType):
        """
        Return instantaneous phase from bandpass filtered velocities on trial specificied by window
        duration.

        Params:
        -------
        source (str)
        windowDur (list of floats)
        trialType (str)
            'avatar', 'avatar0', 'hand', 'hand0'
        """
        raise NotImplementedError("Needs to be fixed.")
        ix = []
        i = 0
        for spec,_ in self.windowsByPart[trialType]:
            if np.isclose(windowDur,spec[1]):
                ix.append(i)
            i += 1
        
        selection = []
        for i in ix:
            try:
                if source=='subject' or source=='s':
                    phases = pickle.load(open('%s/subject_phase_%s_%d.p'%(self.dr,trialType,i),
                                         'rb'))['phases']
                elif source=='template' or source=='t':
                    phases = pickle.load(open('%s/template_phase_%s_%d.p'%(self.dr,trialType,i),
                                         'rb'))['phases']

                phases = [np.vstack(p) for p in phases]
                selection.append(( self.windowsByPart[trialType][i][0],phases ))
            except IOError:
                print "Trial %d in trial type %s not found."%(i,trialType)
        return selection

    def phase_by_window_spec(self,source,windowSpec,trial_type):
        """
        Parameters
        ----------
        source : str
        windowSpec : list
        trial_type : str
        """
        selection = []
        for specix,spec in enumerate(windowSpec):
            ix = self._fetch_windowspec_indices([spec],trial_type,precision=precision)
            
            if len(ix)>0:
                selection.append(( self.windowsByPart[trial_type][ix[0]][0],
                                   self.timeSplitTrials[trial_type][ix[0]],
                                   self.templateSplitTrials[trial_type][ix[0]] ))
                try:
                    if source=='subject' or source=='s':
                        data = pickle.load(open('%s/subject_phase_%s_%d.p'%(self.dr,trial_type,ix[0]),'rb'))
                        phases,vs = data['phases'],data['vs']
                    elif source=='template' or source=='t':
                        data = pickle.load(open('%s/template_phase_%s_%d.p'%(self.dr,trial_type,ix[0]),'rb'))
                        phases,vs = data['phases'],data['vs']
                    
                    phases = [np.vstack(p) for p in phases]
                    selection.append(( self.windowsByPart[trial_type][ix[0]][0],phases ))
                except IOError:
                    print "Trial %d in trial type %s not found."%(ix[0],trial_type)

            # Iterate also through hand0 or avatar0, which contains the other hand.
            if trial_type.isalpha():
                selection += self.phase_by_window_spec(source,
                                                        [windowSpec[specix]],
                                                        trial_type+'0',
                                                        precision=precision)
        return selection

    def filtv_by_window_spec(self,source,windowSpec,trialType,search_all=True):
        """
        Returns:
        --------
        list of twoples (windowSpec, filtv) where filtv is a list of 3 arrays corresponding to each dimension
        """
        raise NotImplementedError()
        ix = self._fetch_windowspec_indices(windowSpec,trialType,precision=precision)
        selection = []

        for i in ix:
            if source=='subject' or source=='s':
                data = pickle.load(open('%s/subject_phase_%s_%d.p'%(self.dr,trialType,i),'rb'))
                phases,vs = data['phases'],data['vs']
            elif source=='template' or source=='t':
                data = pickle.load(open('%s/template_phase_%s_%d.p'%(self.dr,trialType,i),'rb'))
                phases,vs = data['phases'],data['vs']
            else:
                raise Exception

            vs = [np.vstack(p) for p in vs]
            selection.append(( self.windowsByPart[trialType][i][0],vs ))

        if trialType.isalpha() and search_all:
            return selection + self.filtv_by_window_spec(source,windowSpec,trialType+'0',False)

        return selection

    def dphase_by_window_dur(self,windowDur,trialType):
        """
        Difference in phase between subject and template motion.
        """
        raise NotImplementedError
        from misc.angle import mod_angle
        
        subjectPhase = self.phase_by_window_dur('s',windowDur,trialType)
        templatePhase = self.phase_by_window_dur('t',windowDur,trialType)
        dphase = []
        
        for i in xrange(len(subjectPhase)):
            dphase.append(( subjectPhase[i][0], 
                            [mod_angle( s-t ) for s,t in zip(subjectPhase[i][1],templatePhase[i][1])] ))
        if trialType.isalpha():
            return dphase + self.dphase_by_window_dur(windowDur,trialType+'0')
        return dphase

    def dphase_by_window_spec(self,windowSpec,trialType):
        """
        Difference in phase between subject and template motion.
        """
        from misc.angle import mod_angle
        
        subjectPhase = self.phase_by_window_spec('s',windowSpec,trialType)
        templatePhase = self.phase_by_window_spec('t',windowSpec,trialType)
        dphase = []
            
        for i in xrange(len(subjectPhase)):
            dphase.append(( subjectPhase[i][0], 
                            [mod_angle( s-t ) for s,t in zip(subjectPhase[i][1],templatePhase[i][1])] ))
        return dphase

    def pickle_trial_dicts(self,disp=False):
        """
        Put data for analysis into easily accessible pickles. Right now, I extract only visibility
        and hand velocities for AN port data and avatar's motionbuilder files.
        """
        from pipeline import extract_motionbuilder_model2,extract_AN_port
        from utils import match_time

        # Load AN data.
        df = pickle.load(open('%s/%s'%(self.dr,'quickload_an_port_vr.p'),'rb'))['df']
        windowsByPart = window_specs(self.person,self.dr)
        
        # Sort trials into the hand, arm, and avatar trial dictionaries: subjectTrial,
        # templateTrial, hmdTrials.
        subjectTrial,templateTrial,hmdTrials = {},{},{}
        timeSplitTrials,subjectSplitTrials,templateSplitTrials = {},{},{}

        for trialno,part in enumerate(['avatar','avatar0','hand','hand0']):
            if disp:
                print "Processing %s..."%part
            # Select time interval during which the trial happened.
            if part.isalpha():
                visible,invisible = load_visibility(part+'_visibility.txt',self.dr)
            else:
                visible,invisible = load_visibility(part[:-1]+'_visibility_0.txt',self.dr)
            startEnd = [visible[0],visible[-1]]
            
            # Extract template.
            mbT,mbV = extract_motionbuilder_model2(part,startEnd[0],self.modelhandedness[trialno])
            showIx = (mbT>startEnd[0]) & (mbT<startEnd[1])
            templateTrial[part+'T'],templateTrial[part+'V'] = mbT[showIx],mbV[showIx]
            
            # Extract subject from port file.
            anT,anX,anV,anA = extract_AN_port( df,self.modelhandedness[trialno],
                                               rotation_angle=self.rotation[trialno] )
            showIx = (anT>startEnd[0]) & (anT<startEnd[1])
            subjectTrial[part+'T'],subjectTrial[part+'V'] = anT[showIx],anV[0][showIx]
            
            if disp:
                print ("For trial %s, template ends at %s and subject at "+
                        "%s.")%(part,
                                str(templateTrial[part+'T'][-1])[11:],
                                str(subjectTrial[part+'T'][-1])[11:])

            # Put trajectories on the same time samples so we can pipeline our regular computation.
            # Since the AN trial starts after the mbTrial...the offset is positive.
            subjectTrial[part+'V'],subjectTrial[part+'T'] = match_time(subjectTrial[part+'V'],
                   subjectTrial[part+'T'],
                   1/60,
                   offset=(subjectTrial[part+'T'][0]-templateTrial[part+'T'][0]).total_seconds(),
                   use_univariate=True)
            templateTrial[part+'V'],templateTrial[part+'T'] = match_time(templateTrial[part+'V'],
                                                                     templateTrial[part+'T'],
                                                                     1/60,
                                                                     use_univariate=True)
            
            # Separate the different visible trials into separate arrays.
            # Times for when visible/invisible windows start.
            start = np.zeros((len(visible)+len(invisible)),dtype=object)
            start[::2] = visible
            start[1::2] = invisible
            # Units of seconds.
            start = np.array(map(lambda t:t.total_seconds(),np.diff(start)))
            start = np.cumsum(start)
            invisibleStart = start[::2]  # as seconds
            visibleStart = start[1::2]  # as seconds
            
            # When target is invisible, set visibility to 0.
            visibility = np.ones_like(templateTrial[part+'T'])
            for i,j in zip(invisibleStart,visibleStart):
                assert i<j
                visibility[(templateTrial[part+'T']>=i) & (templateTrial[part+'T']<j)] = 0
            if len(visible)<len(invisible):
                visibility[(templateTrial[part+'T']>=invisible[-1])] = 0
            templateTrial[part+'visibility'] = visibility

            timeSplitTrials[part],subjectSplitTrials[part],templateSplitTrials[part] = [],[],[]
            templateSplitTrials[part+'visibility'] = []
            for spec,startendt in windowsByPart[part]:
                startendt = ((startendt[0]-startEnd[0]).total_seconds(),
                             (startendt[1]-startEnd[0]).total_seconds())

                # Save time.
                timeix = (templateTrial[part+'T']<=startendt[1])&(templateTrial[part+'T']>=startendt[0])
                t = templateTrial[part+'T'][timeix]
                timeSplitTrials[part].append(t)

                # Save visibility window.
                templateSplitTrials[part+'visibility'].append( visibility[timeix] )
                
                # Save velocities.
                templateSplitTrials[part].append( templateTrial[part+'V'](t) )
                # Subject sometimes has cutoff window so must reindex time.
                timeix = (subjectTrial[part+'T']<=startendt[1])&(subjectTrial[part+'T']>=startendt[0])
                t = subjectTrial[part+'T'][timeix]
                subjectSplitTrials[part].append( subjectTrial[part+'V'](t) )
            
            # Get the beginning fully visible window. Insert this into the beginning of the list.
            windowsByPart[part].insert(0,((0,0),(0,invisibleStart[0])))
            timeix = (subjectTrial[part+'T']>=0)&(subjectTrial[part+'T']<=invisibleStart[0])
            t = subjectTrial[part+'T'][timeix]
            
            timeSplitTrials[part].insert(0,t)
            subjectSplitTrials[part].insert( 0,subjectTrial[part+'V'](t) )
            templateSplitTrials[part].insert( 0,templateTrial[part+'V'](t) )

            timeix = (templateTrial[part+'T']<=invisibleStart[0])&(templateTrial[part+'T']>=0)
            templateSplitTrials[part+'visibility'].insert( 0,visibility[timeix] )
        
        pickle.dump({'templateTrial':templateTrial,
                     'subjectTrial':subjectTrial,
                     'timeSplitTrials':timeSplitTrials,
                     'templateSplitTrials':templateSplitTrials,
                     'subjectSplitTrials':subjectSplitTrials,
                     'windowsByPart':windowsByPart},
                    open('%s/trial_dictionaries.p'%self.dr,'wb'),-1)

    def pickle_phase(self,trial_types=['avatar','avatar0','hand','hand0']):
        """
        Calculate bandpass filtered phase and pickle.
        """
        from pipeline import pipeline_phase_calc
        
        for part in trial_types:
            nTrials = len(self.windowsByPart[part])  # number of trials for that part

            # Subject.
            toProcess = []
            trialNumbers = []
            for i in xrange(nTrials):
                # Only run process if we have data points. Some trials are missing data points.
                # NOTE: At some point the min length should made to correspond to the min window
                # size in the windowing function for filtering.
                if len(self.timeSplitTrials[part][i])>501:
                    trialNumbers.append(i)
                    toProcess.append( (self.timeSplitTrials[part][i],
                                       (self.subjectSplitTrials[part][i][:,0],
                                        self.subjectSplitTrials[part][i][:,1],
                                        self.subjectSplitTrials[part][i][:,2])) )
                else:
                    print "Ignoring %s trial no %d with windowspec (%1.1f,%1.1f)."%(part,i,
                        self.windowsByPart[part][i][0][0],self.windowsByPart[part][i][0][1])
            pipeline_phase_calc(trajs=toProcess,dr=self.dr,
                                file_names=['subject_phase_%s_%d'%(part,i)
                                            for i in trialNumbers])
            # Template.
            toProcess = []
            trialNumbers = []
            for i in xrange(nTrials):
                if len(self.timeSplitTrials[part][i])>501:
                    trialNumbers.append(i)
                    toProcess.append( (self.timeSplitTrials[part][i],
                                       (self.templateSplitTrials[part][i][:,0],
                                        self.templateSplitTrials[part][i][:,1],
                                        self.templateSplitTrials[part][i][:,2])) )
            pipeline_phase_calc(trajs=toProcess,dr=self.dr,
                                file_names=['template_phase_%s_%d'%(part,i)
                                            for i in trialNumbers])

    def _fetch_windowspec_indices(self,specs,trial_type,precision=None):
        """
        Given a particular trial type and a window specification, return all the indices within that
        trial type that match the given specification.  Options for adjusting the precision for
        matching windows.

        Params
        ------
        trial_type : str
        spec : list of tuples
        
        Returns
        -------
        ix : list of ints
        """
        ix = []
        trialWindows = np.array([w[0] for w in self.windowsByPart[trial_type]])
        i = 0  # counter

        if precision is None:
            for spec in specs:
                ix_ = (np.array(spec)[None,:]==trialWindows).all(1)
                if ix_.any():
                    ix.append( np.where(ix_)[0][0] )
        elif type(precision) is float:
            for spec in specs:
                specDiffs = np.abs( trialWindows-np.array(spec)[None,:] )
                ix_ = (specDiffs<=precision).all(1)
                if ix_.any():
                    ix.append(np.where(ix_)[0][0])
        elif type(precision) is tuple:
            for spec in specs:
                specDiffs = np.abs( trialWindows-np.array(spec)[None,:] )
                ix_ = (specDiffs[:,0]<=precision[0])&(specDiffs[:,1]<=precision[1])
                if ix_.any():
                    ix.append(np.where(ix_)[0][0])
        else:
            raise NotImplementedError("precision type not supported.")

        return ix
# end VRTrial3


class Node(object):
    def __init__(self,name=None,parents=[],children=[]):
        self.name = name
        self.parents = parents
        self.children = children

    def add_child(self,child):
        self.children.append(child)

class Tree(object):
    def __init__(self,nodes):
        """
        Data structure for BVH skeleton hierarchy.

        Attributes:
        -----------
        _nodes (Node)
        nodes
        adjacency
        """
        self._nodes = nodes
        self.nodes = [n.name for n in nodes]
        names = [n.name for n in nodes]
        if len(np.unique(names))<len(names):
            raise Exception("Nodes have duplicate names.")

        self.adjacency = np.zeros((len(nodes),len(nodes)))
        for i,n in enumerate(nodes):
            for c in n.children:
                try:
                    self.adjacency[i,names.index(c)] = 1
                # automatically insert missing nodes (these should all be dangling)
                except ValueError:
                    self.adjacency = np.pad( self.adjacency, ((0,1),(0,1)), mode='constant', constant_values=0)
                    self._nodes.append( Node(c) )
                    names.append(c)

                    self.adjacency[i,names.index(c)] = 1
        
    def print_tree(self):
        print self.adjacency
    
    def parents(self,node):
        """
        Return parents of particular node.

        Returns:
        --------
        parents (list)
            Parents starting from immediate parent and ascending up the tree.
        """
        parents = []
        iloc = self.nodes.index(node)

        while np.any(self.adjacency[:,iloc]):
            iloc = np.where(self.adjacency[:,iloc])[0][0]
            parents.append(self.nodes[iloc])

        return parents

