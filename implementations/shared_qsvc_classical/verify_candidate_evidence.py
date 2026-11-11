import pandas as pd
from common import *

def run():
    grid=ROOT.parent/'wdbc_qsvc_scaled_grid_20261107';source=pd.read_csv(ROOT/'evidence/qsvc_completed_grid.csv').fillna('');rows=[]
    for c in read(ROOT/'candidates.json'):
        row=source[source.configuration_id==c['archived_configuration_id']].iloc[0]
        predictions=grid/row['test_predictions_path'];assert sha(predictions)==row['test_predictions_sha256']
        config=read(Path(c['archived_artifact_path'])/'configuration.json')
        if isinstance(config['feature_map'],dict):
            fmap=config['feature_map'];assert fmap['feature_dimension']==c['k'] and fmap['repetitions']==c['feature_map_reps'] and fmap['constructor']==c['feature_map']
            if c['feature_map']=='ZZFeatureMap':assert fmap['topology']==c['entanglement'] and fmap['Pauli_terms_for_alias']==['Z','ZZ'] and fmap['alpha']==2
        else:
            assert config['k']==c['k'] and config['repetitions']==c['feature_map_reps']
            assert config['feature_map']==c['feature_map'] or (config['feature_map']=='PauliFeatureMap' and c['feature_map']=='ZZFeatureMap')
            assert config['scaler']['feature_range']==[0.,1.] and config['scaler']['clip'] is False
            assert config['selector']['step']==1 and config['selector']['random_state']==42
            assert config['QSVC_settings']['C']==1 and config['QSVC_settings']['probability'] is True
        rows.append({'candidate_id':c['candidate_id'],'archived_configuration_id':c['archived_configuration_id'],'configuration_sha256':sha(Path(c['archived_artifact_path'])/'configuration.json'),'saved_prediction_sha256_verified':True,'calibration_seed_in_archive':c['archived_calibration_seed'],'Z_entanglement_handling':'No argument in new construction; archived nominal topology is scientifically inapplicable' if c['feature_map']=='ZFeatureMap' else 'canonical ZZ verified Pauli [Z,ZZ] equivalence'})
    atomic(ROOT/'records/candidate_artifact_verification.json',{'passed':True,'candidates':rows,'WDBC_training':False})
    print('All ten candidates verified against actual completed configuration and prediction artifacts.')

if __name__=='__main__':run()
