from pathlib import Path
import json
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


class RepositoryInputTests(unittest.TestCase):
    def test_published_scenarios_have_local_inputs(self):
        # These incomplete legacy scenarios are ignored but may still exist locally.
        local_only = {
            'named_roads_peak_hours.sumocfg',
            'named_roads_peak_hours_wide_coverage.sumocfg',
        }
        scenarios = [path for path in (ROOT / 'sumo/scenarios').glob('*.sumocfg')
                     if path.name not in local_only]
        self.assertTrue(scenarios)
        for scenario in scenarios:
            for entry in ET.parse(scenario).findall('./input/*'):
                for value in entry.attrib['value'].split(','):
                    target = (scenario.parent / value.strip()).resolve()
                    with self.subTest(scenario=scenario.name, input=value):
                        self.assertTrue(target.is_file(), str(target))

    def test_simulation_configs_have_local_inputs(self):
        configs = [ROOT / 'src/simulation/xlsx_traci_flow_config.json']
        configs += list((ROOT / 'src/simulation/configs').glob('*.json'))
        for path in configs:
            config = json.loads(path.read_text(encoding='utf-8'))
            for section, key in [('sumo', 'sumocfg_path'), ('sumo', 'route_source_path'), ('xlsx', 'path')]:
                target = (path.parent / config[section][key]).resolve()
                with self.subTest(config=path.name, input=key):
                    self.assertTrue(target.is_file(), str(target))
            scenario = (path.parent / config['sumo']['sumocfg_path']).resolve()
            for entry in ET.parse(scenario).findall('./input/*'):
                for value in entry.attrib['value'].split(','):
                    target = (scenario.parent / value.strip()).resolve()
                    with self.subTest(scenario=scenario.name, input=value):
                        self.assertTrue(target.is_file(), str(target))

    def test_training_backbones_and_reference_tables_exist(self):
        for path in (ROOT / 'configs').glob('*.json'):
            config = json.loads(path.read_text(encoding='utf-8'))
            for key in ('dmfgcrn_checkpoint_path', 'dmfgcrn_conf_path'):
                if key in config:
                    with self.subTest(config=path.name, input=key):
                        self.assertTrue((ROOT / config[key]).is_file())
        for name in ('detector_mapping.csv', 'sensor_node_mapping.csv'):
            self.assertTrue((ROOT / 'datasets/processed/references' / name).is_file())
        self.assertTrue((ROOT / 'datasets/raw/sumo_data/priority_tls_sensors_for_edges.csv').is_file())


if __name__ == '__main__':
    unittest.main()
