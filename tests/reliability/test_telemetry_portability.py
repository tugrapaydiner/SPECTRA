from types import SimpleNamespace
import pytest
import eval.telemetry as t


@pytest.mark.parametrize('exc',[FileNotFoundError('sysfs absent'),PermissionError('sysfs denied'),NotImplementedError('unsupported')])
def test_battery_provider_errors_are_explicit_not_crashes(monkeypatch,exc):
    def fail():raise exc
    monkeypatch.setattr(t,'psutil',SimpleNamespace(sensors_battery=fail))
    record=t.battery_observation()
    assert record['status']=='read_unavailable' and record['level'] is None and record['plugged'] is None
    assert t._read_battery()==(1.,1.) # documented controller fallback, not measured battery


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1.,101.,'bad'])
def test_invalid_battery_values_not_measurements(monkeypatch,value):
    monkeypatch.setattr(t,'psutil',SimpleNamespace(sensors_battery=lambda:SimpleNamespace(percent=value,power_plugged=True)))
    assert t.battery_observation()['level'] is None
    assert t._read_battery()==(1.,1.)


def test_observed_absent_optional_provider(monkeypatch):
    monkeypatch.setattr(t,'psutil',SimpleNamespace(sensors_battery=lambda:SimpleNamespace(percent=25.,power_plugged=False)))
    assert t.battery_observation()=={'level':.25,'plugged':False,'status':'observed'}
    assert t._read_battery()==(.25,0.)
    monkeypatch.setattr(t,'psutil',None)
    assert t.battery_observation()['status']=='provider_unavailable'
