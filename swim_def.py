# Mostly taken fom Alfonso's PID_2_DST repo https://git.km3net.de/alazo/pid_2_dst/-/tree/main/

import km3pipe as kp
import numpy as np
import pandas as pd


class Add_SWIMinfo_neutrino(kp.Module):

    def configure(self):
        # Option for having global verbosity
        self.log.setLevel(self.get("verbosity", default="WARNING"))
        # Register the service to append branches to the pump
        self.require_service("branches")

    def prepare(self):
        self.branches = self.services["branches"]
        # Requiered branches can be now added :
        to_add = [
        'E.mc_trks.type[:,0]',
        'E.run_id',
        'T.sum_mc_evt.livetime_DAQ',
        'E.mc_id',
        'T.sum_mc_nu.cc',
        'E.mc_trks.pos.x[:,0]',
        'E.mc_trks.pos.y[:,0]',
        'E.mc_trks.pos.z[:,0]',
        'E.trks.pos.x[:,0]',
        'E.trks.pos.y[:,0]',
        'E.trks.pos.z[:,0]',
        'E.mc_trks.E[:,0]',
        'E.mc_trks.dir.z[:,0]',
        'T.sum_mc_nu.by',
        'E.trks.E[:,0]',
        'E.trks.len[:,0]',
        'E.trks.dir.z[:,0]',
        'T.feat_Neutrino2020.gandalf_pos_r',
        'T.feat_Neutrino2020.maximumToT_triggerHit',
        'E.trks.lik[:,0]',
        'E.trks.fitinf[:,0,0]',
        'T.feat_Neutrino2020.nTriggerHits',
        'T.feat_Neutrino2020.meanZhitTrig',
        'E.w[:,0]',
        'E.w[:,1]',
        'E.w2list[:,1]',
        'E.w2list[:,5]',
        'T.sum_mc_evt.n_gen',
        'T.sum_mc_evt.E_min_gen',
        'T.sum_mc_evt.E_max_gen',
        'tree.muonscore',
        'tree.trackscore',
        'E.trks.rec_type[:,0]'
        ]
        for b in to_add : self.branches.append(b)

    def process(self, blob):
        # Called for each iteration of the pipeline

        frame = pd.DataFrame(blob['tree'])
        #print(frame.keys())
        #blob['tree']['energy'] = frame['E.mc_trks.E[:,0]']
        blob['tree']['type'] = frame['E.mc_trks.type[:,0]']
        blob['tree']['run_id'] = frame['E.run_id']
        blob['tree']['run_duration'] = frame['T.sum_mc_evt.livetime_DAQ'].astype('float64')
        blob['tree']['mc_id'] = frame['E.mc_id']
        blob['tree']['is_cc'] = frame['T.sum_mc_nu.cc'].astype('bool')
        frame['is_neutrino'] = bool(1)
        blob['tree']['is_neutrino'] = frame['is_neutrino']

        blob['tree']['pos_x_true'] = frame['E.mc_trks.pos.x[:,0]'].astype('float64')
        blob['tree']['pos_y_true'] = frame['E.mc_trks.pos.y[:,0]'].astype('float64')
        blob['tree']['pos_z_true'] = frame['E.mc_trks.pos.z[:,0]'].astype('float64')
        blob['tree']['pos_x_recoJGandalf'] = frame['E.trks.pos.x[:,0]'].astype('float64')
        blob['tree']['pos_y_recoJGandalf'] = frame['E.trks.pos.y[:,0]'].astype('float64')
        blob['tree']['pos_z_recoJGandalf'] = frame['E.trks.pos.z[:,0]'].astype('float64')
        frame['pos_x_recoDusj'] = np.float64(0)
        frame['pos_y_recoDusj'] = np.float64(0)
        frame['pos_z_recoDusj'] = np.float64(0)
        blob['tree']['pos_x_recoDusj'] = frame['pos_x_recoDusj']
        blob['tree']['pos_y_recoDusj'] = frame['pos_y_recoDusj']
        blob['tree']['pos_z_recoDusj'] = frame['pos_z_recoDusj']

        blob['tree']['energy_true'] = frame['E.mc_trks.E[:,0]'].astype('float64')
        blob['tree']['cos_zenith_true'] = np.float64(-1)*frame['E.mc_trks.dir.z[:,0]'].astype('float64')
        blob['tree']['bjorken_y_true'] = frame['T.sum_mc_nu.by'].astype('float64')
        blob['tree']['energy_recoJEnergy'] = frame['E.trks.E[:,0]'].astype('float64')#energy_recoTracklength
        blob['tree']['energy_recoTracklength'] = np.float64(0.25)*frame['E.trks.len[:,0]'].astype('float64')#cos_zenith_recoJGandalf
        blob['tree']['energy_recoRatioEL_JEnergy'] = frame['E.trks.E[:,0]'].astype('float64')/(np.float64(2.0* 6371.0088) *frame['E.trks.len[:,0]'].astype('float64'))
        blob['tree']['energy_recoRatioLE_JEnergy'] = (np.float64(2.0* 6371.0088) *frame['E.trks.len[:,0]'].astype('float64'))/frame['E.trks.E[:,0]'].astype('float64')
        blob['tree']['energy_recoRatioEL_Tracklen'] = (np.float64(0.25)*frame['E.trks.len[:,0]'].astype('float64'))/(np.float64(2.0* 6371.0088) *frame['E.trks.len[:,0]'].astype('float64'))
        blob['tree']['energy_recoRatioLE_Tracklen'] = (np.float64(2.0* 6371.0088) *frame['E.trks.len[:,0]'].astype('float64'))/(np.float64(0.25)*frame['E.trks.len[:,0]'].astype('float64'))
        blob['tree']['cos_zenith_recoJGandalf'] = np.float64(-1)*frame['E.trks.dir.z[:,0]'].astype('float64')
        frame['bjorken_y_recoJGandalf'] = np.float64(0.5)
        blob['tree']['bjorken_y_recoJGandalf'] = frame['bjorken_y_recoJGandalf']
        frame['energy_recoDusj'] = np.float64(0.5)
        frame['cos_zenith_recoDusj'] = np.float64(-0.5)
        frame['bjorken_y_recoDusj'] = np.float64(0.5)
        blob['tree']['energy_recoDusj'] = frame['energy_recoDusj']
        blob['tree']['cos_zenith_recoDusj'] = frame['cos_zenith_recoDusj']
        blob['tree']['bjorken_y_recoDusj'] = frame['bjorken_y_recoDusj']

        blob['tree']['gandalf_pos_r'] = frame['T.feat_Neutrino2020.gandalf_pos_r']
        blob['tree']['maximumToT_triggerHit'] = frame['T.feat_Neutrino2020.maximumToT_triggerHit']
        blob['tree']['gandalf_lik'] = frame['E.trks.lik[:,0]']
        blob['tree']['beta0'] = frame['E.trks.fitinf[:,0,0]']
        blob['tree']['meanZhitTrig'] = frame['T.feat_Neutrino2020.meanZhitTrig']
        blob['tree']['nTriggerHits'] = frame['T.feat_Neutrino2020.nTriggerHits']

        blob['tree']['w1'] = frame['E.w[:,0]']
        blob['tree']['w2'] = frame['E.w[:,1]']*np.float64(365.25*24.0*60.0*60.0)
        blob['tree']['int_len'] = frame['E.w2list[:,5]']
        blob['tree']['ngen'] = frame['T.sum_mc_evt.n_gen'].astype('float64')
        blob['tree']['EG'] = frame['E.w2list[:,1]']
        blob['tree']['E_min_gen'] = frame['T.sum_mc_evt.E_min_gen'].astype('float64')
        blob['tree']['E_max_gen'] = frame['T.sum_mc_evt.E_max_gen'].astype('float64')

        frame['weight_one_year'] = np.float64(0)
        blob['tree']['weight_one_year'] = frame['weight_one_year']
        blob['tree']['antimu_proba_bkg'] = frame['tree.muonscore']
        blob['tree']['pid_proba_track'] = frame['tree.trackscore']
        blob['tree']['rectype_JGandalf'] = frame['E.trks.rec_type[:,0]']


        return blob

    def finish(self):
        # Called once when the pipeline is closing
        pass


class Add_SWIMinfo_muon(kp.Module):

    def configure(self):
        # Option for having global verbosity
        self.log.setLevel(self.get("verbosity", default="WARNING"))
        # Register the service to append branches to the pump
        self.require_service("branches")
        self.totaltime = self.get("totaltime", default=460.7983564814815*60*60*24)

    def prepare(self):
        self.branches = self.services["branches"]
        # Requiered branches can be now added :
        to_add = [
        'E.mc_trks.type[:,0]',
        'E.run_id',
        'T.sum_mc_evt.livetime_DAQ',
        'E.mc_id',
        'E.mc_trks.pos.x[:,0]',
        'E.mc_trks.pos.y[:,0]',
        'E.mc_trks.pos.z[:,0]',
        'E.trks.pos.x[:,0]',
        'E.trks.pos.y[:,0]',
        'E.trks.pos.z[:,0]',
        'E.mc_trks.E[:,0]',
        'E.mc_trks.dir.z[:,0]',
        'E.trks.E[:,0]',
        'E.trks.len[:,0]',
        'E.trks.dir.z[:,0]',
        'T.feat_Neutrino2020.gandalf_pos_r',
        'T.feat_Neutrino2020.maximumToT_triggerHit',
        'E.trks.lik[:,0]',
        'E.trks.fitinf[:,0,0]',
        'T.feat_Neutrino2020.nTriggerHits',
        'T.feat_Neutrino2020.meanZhitTrig',
        'tree.muonscore',
        'tree.trackscore',
        'T.sum_mc_evt.livetime_DAQ',
        'T.sum_mc_evt.livetime_sim',
        'E.trks.rec_type[:,0]',
        ]
        for b in to_add : self.branches.append(b)

    def process(self, blob):
        # Called for each iteration of the pipeline

        frame = pd.DataFrame(blob['tree'])
        #print(frame.keys())

        blob['tree']['type'] = frame['E.mc_trks.type[:,0]']
        blob['tree']['run_id'] = frame['E.run_id']
        blob['tree']['run_duration'] = frame['T.sum_mc_evt.livetime_DAQ'].astype('float64')
        blob['tree']['mc_id'] = frame['E.mc_id']
        frame['is_cc'] = bool(0)
        blob['tree']['is_cc'] = frame['is_cc']
        frame['is_neutrino'] = bool(0)
        blob['tree']['is_neutrino'] = frame['is_neutrino']

        blob['tree']['pos_x_true'] = frame['E.mc_trks.pos.x[:,0]'].astype('float64')
        blob['tree']['pos_y_true'] = frame['E.mc_trks.pos.y[:,0]'].astype('float64')
        blob['tree']['pos_z_true'] = frame['E.mc_trks.pos.z[:,0]'].astype('float64')
        blob['tree']['pos_x_recoJGandalf'] = frame['E.trks.pos.x[:,0]'].astype('float64')
        blob['tree']['pos_y_recoJGandalf'] = frame['E.trks.pos.y[:,0]'].astype('float64')
        blob['tree']['pos_z_recoJGandalf'] = frame['E.trks.pos.z[:,0]'].astype('float64')
        frame['pos_x_recoDusj'] = np.float64(0)
        frame['pos_y_recoDusj'] = np.float64(0)
        frame['pos_z_recoDusj'] = np.float64(0)
        blob['tree']['pos_x_recoDusj'] = frame['pos_x_recoDusj']
        blob['tree']['pos_y_recoDusj'] = frame['pos_y_recoDusj']
        blob['tree']['pos_z_recoDusj'] = frame['pos_z_recoDusj']

        blob['tree']['energy_true'] = frame['E.mc_trks.E[:,0]'].astype('float64')
        blob['tree']['cos_zenith_true'] = np.float64(-1)*frame['E.mc_trks.dir.z[:,0]'].astype('float64')
        frame['bjorken_y_true'] = np.float64(0.5)
        blob['tree']['bjorken_y_true'] = frame['bjorken_y_true'].astype('float64')
        blob['tree']['energy_recoJEnergy'] = frame['E.trks.E[:,0]'].astype('float64')#energy_recoTracklength
        blob['tree']['energy_recoTracklength'] = np.float64(0.25)*frame['E.trks.len[:,0]'].astype('float64')#cos_zenith_recoJGandalf
        blob['tree']['energy_recoRatioEL_JEnergy'] = frame['E.trks.E[:,0]'].astype('float64')/(np.float64(2.0* 6371.0088) *frame['E.trks.len[:,0]'].astype('float64'))
        blob['tree']['energy_recoRatioLE_JEnergy'] = (np.float64(2.0* 6371.0088) *frame['E.trks.len[:,0]'].astype('float64'))/frame['E.trks.E[:,0]'].astype('float64')
        blob['tree']['energy_recoRatioEL_Tracklen'] = (np.float64(0.25)*frame['E.trks.len[:,0]'].astype('float64'))/(np.float64(2.0* 6371.0088) *frame['E.trks.len[:,0]'].astype('float64'))
        blob['tree']['energy_recoRatioLE_Tracklen'] = (np.float64(2.0* 6371.0088) *frame['E.trks.len[:,0]'].astype('float64'))/(np.float64(0.25)*frame['E.trks.len[:,0]'].astype('float64'))
        blob['tree']['cos_zenith_recoJGandalf'] = np.float64(-1)*frame['E.trks.dir.z[:,0]'].astype('float64')
        frame['bjorken_y_recoJGandalf'] = np.float64(0.5)
        blob['tree']['bjorken_y_recoJGandalf'] = frame['bjorken_y_recoJGandalf']
        frame['energy_recoDusj'] = np.float64(0.5)
        frame['cos_zenith_recoDusj'] = np.float64(-0.5)
        frame['bjorken_y_recoDusj'] = np.float64(0.5)
        blob['tree']['energy_recoDusj'] = frame['energy_recoDusj'].astype('float64')
        blob['tree']['cos_zenith_recoDusj'] = frame['cos_zenith_recoDusj'].astype('float64')
        blob['tree']['bjorken_y_recoDusj'] = frame['bjorken_y_recoDusj'].astype('float64')

        blob['tree']['gandalf_pos_r'] = frame['T.feat_Neutrino2020.gandalf_pos_r']
        blob['tree']['maximumToT_triggerHit'] = frame['T.feat_Neutrino2020.maximumToT_triggerHit']
        blob['tree']['gandalf_lik'] = frame['E.trks.lik[:,0]']
        blob['tree']['beta0'] = frame['E.trks.fitinf[:,0,0]']
        blob['tree']['meanZhitTrig'] = frame['T.feat_Neutrino2020.meanZhitTrig']
        blob['tree']['nTriggerHits'] = frame['T.feat_Neutrino2020.nTriggerHits']

        frame['w2'] = np.float64(0)
        frame['w1'] = np.float64(0)
        frame['int_len'] = np.float64(-1)
        frame['ngen'] = np.float64(-1)
        frame['EG'] = np.float64(-1)
        frame['E_min_gen'] = np.float64(-1)
        frame['E_max_gen'] = np.float64(-1)


        blob['tree']['w1'] = frame['w1']
        blob['tree']['w2'] = frame['w2']
        blob['tree']['int_len'] = frame['int_len']
        blob['tree']['ngen'] = frame['ngen']
        blob['tree']['EG'] = frame['EG']
        blob['tree']['E_min_gen'] = frame['E_min_gen']
        blob['tree']['E_max_gen'] = frame['E_max_gen']

        frame['weight_one_year'] = frame['T.sum_mc_evt.livetime_DAQ']*np.float64(3.15576e7)/(frame['T.sum_mc_evt.livetime_sim']*np.float64(self.totaltime))
        blob['tree']['weight_one_year'] = frame['weight_one_year'].astype('float64')
        blob['tree']['antimu_proba_bkg'] = frame['tree.muonscore']
        blob['tree']['pid_proba_track'] = frame['tree.trackscore']
        blob['tree']['rectype_JGandalf'] = frame['E.trks.rec_type[:,0]']


        return blob

    def finish(self):
        # Called once when the pipeline is closing
        pass

class Add_SWIMinfo_JSHF(kp.Module):

    def configure(self):
        # Option for having global verbosity
        self.log.setLevel(self.get("verbosity", default="WARNING"))
        # Register the service to append branches to the pump
        self.require_service("branches")

    def prepare(self):
        self.branches = self.services["branches"]
        # Requiered branches can be now added :
        to_add = [
            'E.trks.pos.x[:,1]',
            'E.trks.pos.y[:,1]',
            'E.trks.pos.z[:,1]',
            'E.trks.dir.x[:,1]',
            'E.trks.dir.y[:,1]',
            'E.trks.dir.z[:,1]',
            'E.trks.E[:,1]',
            'E.trks.len[:,1]',
            'E.trks.len[:,0]',
            'E.trks.lik[:,1]',
            'E.trks.t[:,1]',
            'E.trks.rec_type[:,1]'
        ]
        for b in to_add : self.branches.append(b)

    def process(self, blob):
        # Called for each iteration of the pipeline

        frame = pd.DataFrame(blob['tree'])
        #print(frame.keys())
        blob['tree']['JShower_lik'] = frame['E.trks.lik[:,1]']

        blob['tree']['pos_x_recoJShower'] = frame['E.trks.pos.x[:,1]'].astype('float64')
        blob['tree']['pos_y_recoJShower'] = frame['E.trks.pos.y[:,1]'].astype('float64')
        blob['tree']['pos_z_recoJShower'] = frame['E.trks.pos.z[:,1]'].astype('float64')

        blob['tree']['energy_recoJShower'] = frame['E.trks.E[:,1]'].astype('float64')
        blob['tree']['energy_recoRatioEL_JShower'] = frame['E.trks.E[:,1]'].astype('float64')/(np.float64(2.0* 6371.0088) *frame['E.trks.len[:,0]'].astype('float64'))
        blob['tree']['energy_recoRatioLE_JShower'] = (np.float64(2.0* 6371.0088) *frame['E.trks.len[:,0]'].astype('float64'))/frame['E.trks.E[:,1]'].astype('float64')
        blob['tree']['cos_zenith_recoJShower'] = np.float64(-1)*frame['E.trks.dir.z[:,1]'].astype('float64')
        frame['bjorken_y_recoJShower'] = np.float64(0.5)
        blob['tree']['bjorken_y_recoJShower'] = frame['bjorken_y_recoJShower'].astype('float64')
        blob['tree']['rectype_JShower'] = frame['E.trks.rec_type[:,1]']



        return blob

    def finish(self):
        # Called once when the pipeline is closing
        pass

class Add_SWIMinfo_DNN(kp.Module):

    def configure(self):
        # Option for having global verbosity
        self.log.setLevel(self.get("verbosity", default="WARNING"))
        # Register the service to append branches to the pump
        self.require_service("branches")

    def prepare(self):
        self.branches = self.services["branches"]
        # Requiered branches can be now added :
        to_add = [
        'DNN.energy_recoDNN',
        'E.trks.E[:,0]',
        'E.trks.len[:,0]',
        ]
        for b in to_add : self.branches.append(b)

    def process(self, blob):
        # Called for each iteration of the pipeline

        frame = pd.DataFrame(blob['tree'])
        #print(frame.keys())
        blob['tree']['energy_recoDNN'] = frame['DNN.energy_recoDNN'].astype('float64')
        blob['tree']['energy_recoRatioEL_DNN'] = frame['DNN.energy_recoDNN'].astype('float64')/(np.float64(2.0* 6371.0088) *frame['E.trks.len[:,0]'].astype('float64'))
        blob['tree']['energy_recoRatioLE_DNN'] = (np.float64(2.0* 6371.0088) *frame['E.trks.len[:,0]'].astype('float64'))/frame['DNN.energy_recoDNN'].astype('float64')


        return blob

    def finish(self):
        # Called once when the pipeline is closing
        pass

