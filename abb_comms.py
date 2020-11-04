from requests.auth import HTTPDigestAuth
from requests.exceptions import RequestException
from requests import Session
import ast
import time
import json
import math


# Address used to organize ET elements
namespace = '{http://www.w3.org/1999/xhtml}'


class RWS:
    """Class for communicating with RobotWare through Robot Web Services (ABB's Rest API).
    Most of the functions are mainly aimed at laboratory work at the University of Stavanger,
    but may hopefully prove useful otherwise as well.
    """

    def __init__(self, base_url, username='Default User', password='robotics'):
        # TODO create check to see if connection worked
        self.base_url = base_url
        self.username = username
        self.password = password
        self.session = Session()  # create persistent HTTP communication
        self.session.auth = HTTPDigestAuth(self.username, self.password)
        self.connected = self.test_connection()

    def test_connection(self):
        try:
            r = self.session.get(self.base_url + "/rw/rapid/execution?json=1", timeout=0.2)
        except RequestException as e:
            return False
        return True

    def get_joints(self):
        ''' return array of joint positions '''
        if self.connected:
            # raise Exception("Not implemented with real robot yet")
            resp = self.session.get(self.base_url + '/rw/motionsystem/mechunits/ROB_1/jointtarget?json=1')  # &json=1
            json_string = resp.text
            _dict = json.loads(json_string)
            data = _dict["_embedded"]["_state"][0]
            jnts = [data['rax_1'], data['rax_2'], data['rax_3'], data['rax_4'], data['rax_5'], data['rax_6']]
            jnts = [float(i) for i in jnts]
        else:
            jnts = [0, 0, 0, 0, 30, 0]
        return jnts

    def get_tool_position(self, toolname, wobjname='wobj0'):
        """ Gets translational and rotational of the UiS tool 'tGripper'
            with respect to the work object 'wobjTableN'.
        """
        if self.connected:
            resp = self.session.get(
                self.base_url + '/rw/motionsystem/mechunits/ROB_1/robtarget/?tool=' + toolname + '&wobj=' + wobjname + '&json=1')
            # raise Exception("Not implemented with real robot yet")
            json_string = resp.text
            _dict = json.loads(json_string)
            data = _dict["_embedded"]["_state"][0]
            trans = [data["x"], data["y"], data["z"]]
            trans = [float(i) for i in trans]
            rot = [data["q1"], data["q2"], data["q3"], data["q4"]]
            rot = [float(i) for i in rot]
            cfg = [data["cf1"], data["cf4"], data["cf6"], data["cfx"]]
            cfg = [float(i) for i in cfg]
            return trans, rot, cfg
        else:
            return [0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]

    def get_file(self, path_from_home):
        ''' Return file contents inside IRC5 controller. '''
        # first, lets test some of the more generic methods
        url = self.base_url + '/fileservice/$home/' + path_from_home
        # retreive a file that already exists
        resp = self.session.get(url)
        if resp.status_code == 200:
            print('file found')
            return resp.text
        else:
            print('Could not find file')
            return None

    def file_transfer_from_file(self, destination_path, source_path):
        ''' send file to IRC5 Home directory
                destination_path: relative path where the file will end up in the IRC5 controller.
                    --> this path is relative to the HOME folder in the IRC5
                    --> Filename must be included (e.g waam_autogen/waam_autogen_test.mod)
                upload_file_path: path the the file that is to be uploaded
        '''
        url = self.base_url + '/fileservice/$home/' + destination_path

        with open(source_path, 'rb') as f:
            resp = self.session.put(url=url, data=f)

        if resp.status_code == 200 or resp.status_code == 201:
            print("File Transfer Successful")
            return resp
        else:
            print("File transfer failed")
            assert False, "File transfer failed"

    def file_transfer_from_string(self, path_from_home, input_str):
        ''' Creates a file in the IRC5 controller and inputs the string data into it
            destination_path: relative path where the file will end up in the IRC5 controller.
                    --> this path is relative to the HOME folder in the IRC5
                    --> Filename must be included (e.g waam_autogen/waam_autogen_test.mod)
            input_str: string that will form the contents of the desination file.
                    --> Include formating in string (i.e. \n \t etc)
            '''
        url = self.base_url + '/fileservice/$home/' + path_from_home

        # with open(upload_file_path, 'rb') as f:
        resp = self.session.put(url=url, data=input_str)

        if resp.status_code == 200 or resp.status_code == 201:
            print("File Transfer Successful")
            return resp
        else:
            print("File transfer failed")
            assert False, "File transfer failed"

    def set_rapid_variable(self, var, value):
        """Sets the value of any RAPID variable.
        Unless the variable is of type 'num', 'value' has to be a string.
        """

        payload = {'value': value}
        resp = self.session.post(self.base_url + '/rw/rapid/symbol/data/RAPID/T_ROB1/' + var + '?action=set',
                                 data=payload)
        return resp

    def get_io_signal(self, var):
        """Gets the raw value of any IO signal.
        """
        resp = self.session.get(self.base_url + '/rw/iosystem/signals/DeviceNet/ioFronius1/' + var + '?json=1')
        json_string = resp.text
        _dict = json.loads(json_string)
        value = _dict['_embedded']['_state'][0]['lvalue']
        return value

    def get_rapid_variable(self, var):
        """Gets the raw value of any RAPID variable.
        """
        resp = self.session.get(self.base_url + '/rw/rapid/symbol/data/RAPID/T_ROB1/' + var + ';value?json=1')
        json_string = resp.text
        _dict = json.loads(json_string)
        value = _dict["_embedded"]["_state"][0]["value"]
        return value

    def get_robtarget_variables(self, var):
        """Gets both translational and rotational data from robtarget.
        """
        resp = self.session.get(self.base_url + '/rw/rapid/symbol/data/RAPID/T_ROB1/' + var + ';value?json=1')
        json_string = resp.text
        _dict = json.loads(json_string)
        data = _dict["_embedded"]["_state"][0]["value"]
        data_list = ast.literal_eval(data)  # Convert the pure string from data to list
        trans = data_list[0]  # Get x,y,z from robtarget relative to work object (table)
        rot = data_list[1]  # Get orientation of robtarget
        cfg = data_list[2]
        return trans, rot, cfg

    def get_wobj(self, var="wobj0"):
        """Gets both translational and rotational data from a wobj of specified name.
        """
        resp = self.session.get(self.base_url + '/rw/rapid/symbol/data/RAPID/T_ROB1/' + var + ';value?json=1')
        json_string = resp.text
        _dict = json.loads(json_string)
        data = _dict["_embedded"]["_state"][0]["value"]
        data = data.replace("FALSE", "False")
        data = data.replace("TRUE", "True")
        data_list = ast.literal_eval(data)  # Convert the pure string from data to list
        trans = data_list[3][0]  # Get x,y,z from robtarget relative to work object (table)
        rot = data_list[3][1]  # Get orientation of robtarget
        return trans, rot

        # resp = self.session.get(self.base_url + '/rw/rapid/symbol/data/RAPID/T_ROB1/' + var + ';value?json=1')
        # json_string = resp.text
        # _dict = json.loads(json_string)
        # value = _dict["_embedded"]["_state"][0]["value"]

    def set_jointtarget_values(self, var, jnts):
        """Sets the j1-j6 for a given jointtarget
        """
        resp = self.set_rapid_variable(var,
                                       "[[" + ','.join([str(s) for s in jnts]) + "], [ 9E9, 9E9,9E9, 9E9, 9E9, 9E9]]")
        return resp

    def set_robtarget_translation(self, var, trans):
        """Sets the translational data of a robtarget variable in RAPID.
        """
        _trans, _rot, _cfg = self.get_robtarget_variables(var)
        dat = "[[" + ','.join([str(s) for s in trans]) + "],[" + ','.join(str(s) for s in _rot) + "],[" + ','.join(
            str(s) for s in _cfg) + "],[9E+9,9E+9,9E+9,9E+9,9E+9,9E+9]]"
        resp = self.set_rapid_variable(var, dat)
        return resp

    def set_robtarget_rotation_quaternion(self, var, rotation_quaternion):
        """Updates the orientation of a robtarget variable in RAPID by a Quaternion.
        """
        _trans, _rot, _cfg = self.get_robtarget_variables(var)
        dat = "[[" + ','.join([str(s) for s in _trans]) + "],[" + ','.join(
            str(s) for s in rotation_quaternion) + "],[" + ','.join(
            str(s) for s in _cfg) + "],[9E+9,9E+9,9E+9,9E+9,9E+9,9E+9]]"
        resp = self.set_rapid_variable(var, dat)
        return resp

    def set_robtarget_cfg(self, var, cfg_data):
        """Updates the orientation of a robtarget variable in RAPID by a Quaternion.
        """
        _trans, _rot, _cfg = self.get_robtarget_variables(var)
        dat = "[[" + ','.join([str(s) for s in _trans]) + "],[" + ','.join(str(s) for s in _rot) + "],[" + ','.join(
            str(s) for s in cfg_data) + "],[9E+9,9E+9,9E+9,9E+9,9E+9,9E+9]]"
        resp = self.set_rapid_variable(var, dat)
        return resp

    def wait_for_rapid(self, var='ready_flag'):
        """Waits for robot to complete RAPID instructions
        until boolean variable in RAPID is set to 'TRUE'.
        Default variable name is 'ready_flag', but others may be used.
        """

        while self.get_rapid_variable(var) == "FALSE" and self.is_running():
            time.sleep(0.1)
        self.set_rapid_variable(var, "FALSE")

    def set_rapid_array(self, var, value):
        """Sets the values of a RAPID array by sending a list from Python.
        """

        # TODO: Check if array must be same size in RAPID and Python
        self.set_rapid_variable(var, "[" + ','.join([str(s) for s in value]) + "]")

    def reset_pp(self):
        """Resets the program pointer to main procedure in RAPID.
        """

        resp = self.session.post(self.base_url + '/rw/rapid/execution?action=resetpp')
        if resp.status_code == 204:
            print('Program pointer reset to main')
        else:
            print('Could not reset program pointer to main')

    def request_mastership(self):
        resp = self.session.post(self.base_url + '/rw/mastership')

    def release_mastership(self):
        resp = self.session.post(self.base_url + '/rw/mastership?action=release')

    def request_rmmp(self):

        # get RMMP state
        # resp = self.session.get(self.base_url + "/users/rmmp/?json=1")
        # json_string = resp.text
        # _dict = json.loads(json_string)
        # uid = _dict["_embedded"]["_state"][0]['userid']
        # priv = _dict["_embedded"]["_state"][0]['privilege']

        # requiest RMMP
        resp = self.session.post(self.base_url + '/users/rmmp', data={'privilege': 'modify'})
        priv = 'pending modify'
        while priv == 'pending modify':  # enter loop to poll the teach pendant ( none|pending modify|modify|exec )
            time.sleep(1)
            self.session.get(
                self.base_url + "/users/rmmp/poll?json=1")  # poll the teach pendant to keep the grant dialog alive
            resp = self.session.get(self.base_url + "/users/rmmp/?json=1")
            json_string = resp.text
            _dict = json.loads(json_string)
            # uid = _dict["_embedded"]["_state"][0]['userid']
            priv = _dict["_embedded"]["_state"][0]['privilege']

            # grant RMMP
        # payload = {'uid': '0', 'privilege': 'deny'} # privilege={modify|exec|deny}
        # resp_grant = self.session.post(self.base_url + "/users/rmmp?action=set", data=payload)
        # resp = self.session.post(self.base_url + '/rw/rapid/symbol/data/RAPID/T_ROB1/' + var + '?action=set',  data=payload)
        # http://localhost

        return resp

    def cancel_rmmp(self):
        resp = self.session.post(self.base_url + '/users/rmmp?action=cancel')

    def motors_on(self):
        """Turns the robot's motors on.
        Operation mode has to be AUTO.
        """

        payload = {'ctrl-state': 'motoron'}
        resp = self.session.post(self.base_url + "/rw/panel/ctrlstate?action=setctrlstate", data=payload)

        if resp.status_code == 204:
            print("Robot motors turned on")
        else:
            print("Could not turn on motors. The controller might be in manual mode")

    def motors_off(self):
        """Turns the robot's motors off.
        """

        payload = {'ctrl-state': 'motoroff'}
        resp = self.session.post(self.base_url + "/rw/panel/ctrlstate?action=setctrlstate", data=payload)

        if resp.status_code == 204:
            print("Robot motors turned off")
        else:
            print("Could not turn off motors")

    def start_RAPID(self):
        """Resets program pointer to main procedure in RAPID and starts RAPID execution.
        """

        self.reset_pp()
        payload = {'regain': 'continue', 'execmode': 'continue', 'cycle': 'once', 'condition': 'none',
                   'stopatbp': 'disabled', 'alltaskbytsp': 'false'}
        resp = self.session.post(self.base_url + "/rw/rapid/execution?action=start", data=payload)
        if resp.status_code == 204:
            print("RAPID execution started from main")
        else:
            opmode = self.get_operation_mode()
            ctrlstate = self.get_controller_state()

            print(f"""
            Could not start RAPID. Possible causes:
            * Operating mode might not be AUTO. Current opmode: {opmode}.
            * Motors might be turned off. Current ctrlstate: {ctrlstate}.
            * RAPID might have write access. 
            """)

    def stop_RAPID(self):
        """Stops RAPID execution.
        """

        payload = {'stopmode': 'stop', 'usetsp': 'normal'}
        resp = self.session.post(self.base_url + "/rw/rapid/execution?action=stop", data=payload)
        if resp.status_code == 204:
            print('RAPID execution stopped')
        else:
            print('Could not stop RAPID execution')

    def get_execution_state(self):
        """Gets the execution state of the controller.
        """

        resp = self.session.get(self.base_url + "/rw/rapid/execution?json=1")
        json_string = resp.text
        _dict = json.loads(json_string)
        data = _dict["_embedded"]["_state"][0]["ctrlexecstate"]
        return data

    def is_running(self):
        """Checks the execution state of the controller and
        """

        execution_state = self.get_execution_state()
        if execution_state == "running":
            return True
        else:
            return False

    def get_operation_mode(self):
        """Gets the operation mode of the controller.
        """

        resp = self.session.get(self.base_url + "/rw/panel/opmode?json=1")
        json_string = resp.text
        _dict = json.loads(json_string)
        data = _dict["_embedded"]["_state"][0]["opmode"]
        return data

    def get_controller_state(self):
        """Gets the controller state.
        """

        resp = self.session.get(self.base_url + "/rw/panel/ctrlstate?json=1")
        json_string = resp.text
        _dict = json.loads(json_string)
        data = _dict["_embedded"]["_state"][0]["ctrlstate"]
        return data

    def set_speed_ratio(self, speed_ratio):
        """Sets the speed ratio of the controller.
        """

        if not 0 < speed_ratio <= 100:
            print("You have entered a false speed ratio value! Try again.")
            return

        payload = {'speed-ratio': speed_ratio}
        resp = self.session.post(self.base_url + "/rw/panel/speedratio?action=setspeedratio", data=payload)
        if resp.status_code == 204:
            print(f'Set speed ratio to {speed_ratio}%')
        else:
            print('Could not set speed ratio!')

    def set_zonedata(self, var, zonedata):
        """Sets the zonedata of a zonedata variable in RAPID.
        """

        if zonedata not in ['fine', 0, 1, 5, 10, 20, 30, 40, 50, 60, 80, 100, 150, 200]:
            print("You have entered false zonedata! Please try again")
            return
        else:
            if zonedata in [10, 20, 30, 40, 50, 60, 80, 100, 150, 200]:
                value = f'[FALSE, {zonedata}, {zonedata * 1.5}, {zonedata * 1.5}, {zonedata * 0.15}, ' \
                        f'{zonedata * 1.5}, {zonedata * 0.15}]'
            elif zonedata == 0:
                value = f'[FALSE, {zonedata + 0.3}, {zonedata + 0.3}, {zonedata + 0.3}, {zonedata + 0.03}, ' \
                        f'{zonedata + 0.3}, {zonedata + 0.03}]'
            elif zonedata == 1:
                value = f'[FALSE, {zonedata}, {zonedata}, {zonedata}, {zonedata * 0.1}, {zonedata}, {zonedata * 0.1}]'
            elif zonedata == 5:
                value = f'[FALSE, {zonedata}, {zonedata * 1.6}, {zonedata * 1.6}, {zonedata * 0.16}, ' \
                        f'{zonedata * 1.6}, {zonedata * 0.16}]'
            else:  # zonedata == 'fine':
                value = f'[TRUE, {0}, {0}, {0}, {0}, {0}, {0}]'

        resp = self.set_rapid_variable(var, value)
        if resp.status_code == 204:
            print(f'Set \"{var}\" zonedata to z{zonedata}')
        else:
            print('Could not set zonedata! Check that the variable name is correct')

    def set_speeddata(self, var, speeddata):
        """Sets the speeddata of a speeddata variable in RAPID.
        """

        resp = self.set_rapid_variable(var, f'[{speeddata},500,5000,1000]')
        if resp.status_code == 204:
            print(f'Set \"{var}\" speeddata to v{speeddata}')
        else:
            print('Could not set speeddata. Check that the variable name is correct')

    def write_to_module(self, module_name, data):  # , var, speeddata):
        ''' Completely overwrite the text in a module. Task is hardcoded for now, but easily added as an unput'''
        payload = {'text': data}
        # resp = self.session.post("http://localhost/rw/rapid/modules/" + module_name + "?task=T_ROB1&action=set-module-text", data=payload)
        resp = self.session.post(
            self.base_url + "/rw/rapid/modules/" + module_name + "?task=T_ROB1&action=set-module-text", data=payload)
        return resp

        # resp = self.session.post(self.base_url + '/rw/rapid/symbol/data/RAPID/T_ROB1/' + var + '?action=set', data=payload)


if __name__ == "__main__":
    conn = RWS('http://192.168.2.18')

    resp = conn.get_operation_mode()
    print(resp)
    tmp = conn.request_rmmp()
    tmp = conn.request_mastership()
    # cJnts = conn.get_joints()
    # print(cJnts)
    resp = conn.set_robtarget_translation("repair_scan_pos", [100, 0, 0])
    print(resp)

    # data = conn.get_io_signal('aoFr1Power')
    # print(data)
    #
    # data = conn.get_rapid_variable('wd1')

'''
    def get_gripper_position(self):
        """Gets translational and rotational of the UiS tool 'tGripper'
        with respect to the work object 'wobjTableN'.
        """

        resp = self.session.get(self.base_url +
                                '/rw/motionsystem/mechunits/ROB_1/robtarget/'
                                '?tool=tGripper&wobj=wobjTableN&coordinate=Wobj&json=1')
        json_string = resp.text
        _dict = json.loads(json_string)
        data = _dict["_embedded"]["_state"][0]
        trans = [data["x"], data["y"], data["z"]]
        trans = [float(i) for i in trans]
        rot = [data["q1"], data["q2"], data["q3"], data["q4"]]
        rot = [float(i) for i in rot] 
        return trans, rot

    def get_gripper_height(self):
        """Extracts only the height from gripper position.
        (See get_gripper_position)
        """ 
        trans, rot = self.get_gripper_position()
        height = trans[2]

        return height
 def set_robtarget_rotation_z_degrees(self, var, rotation_z_degrees):
        """Updates the orientation of a robtarget variable
        in RAPID by rotation about the z-axis in degrees.
        """

        rot = z_degrees_to_quaternion(rotation_z_degrees)

        trans, _rot, cfg = self.get_robtarget_variables(var)

        self.set_rapid_variable(var, "[[" + ','.join(
            [str(s) for s in trans]) + "],[" + ','.join(str(s) for s in rot) + "],[-1,0,0,0],[9E+9,9E+9,9E+9,9E+9,"
                                                                               "9E+9,9E+9]]")

    def send_puck(self, puck_xyz, puck_angle, rotation_z=0, forward_grip=True):
        """Sets gripper angle, camera offset and puck target values chosen.
        If collision check, the variable rotation_z and forward grip may be updated
        """
        rotation_angle = puck_angle - rotation_z

        self.set_rapid_variable("gripper_angle", rotation_z)
        offset_x, offset_y = gripper_camera_offset(rotation_z)
        if forward_grip:
            self.set_rapid_array("gripper_camera_offset", (offset_x, offset_y))
        else:
            self.set_rapid_array("gripper_camera_offset", (-offset_x, -offset_y))
        self.set_robtarget_translation("puck_target", puck_xyz)
        self.set_rapid_variable("puck_angle", rotation_angle)

def quaternion_to_radians(quaternion):
    """Convert a Quaternion to a rotation about the z-axis in degrees.
    """
    w, x, y, z = quaternion
    t1 = +2.0 * (w * z + x * y)
    t2 = +1.0 - 2.0 * (y * y + z * z)
    rotation_z = math.atan2(t1, t2)

    return rotation_z


def z_degrees_to_quaternion(rotation_z_degrees):
    """Convert a rotation about the z-axis in degrees to Quaternion.
    """
    roll = math.pi
    pitch = 0
    yaw = math.radians(rotation_z_degrees)

    qw = math.cos(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) + math.sin(roll / 2) * math.sin(
        pitch / 2) * math.sin(yaw / 2)
    qx = math.sin(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) - math.cos(roll / 2) * math.sin(
        pitch / 2) * math.sin(yaw / 2)
    qy = math.cos(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2) + math.sin(roll / 2) * math.cos(
        pitch / 2) * math.sin(yaw / 2)
    qz = math.cos(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2) - math.sin(roll / 2) * math.sin(
        pitch / 2) * math.cos(yaw / 2)

    return [qw, qx, qy, qz]


def gripper_camera_offset(rot):
    """Finds the offset between the camera and the gripper by using the gripper's orientation.
    Input must be Quaternion or rotation about the z-axis in degrees.
    """

    r = 55  # Distance between gripper and camera

    # Check if input is quaternion
    if isinstance(rot, tuple):
        if len(rot) == 4 and (isinstance(rot[0], int) or isinstance(rot[0], float)):
            rotation_z_radians = quaternion_to_radians(rot)
        else:
            return
    else:
        # If input is not Quaternion, it should be int or float (an angle)

        rotation_z_degrees = rot

    offset_x = r * math.cos(math.radians(rotation_z_degrees))
    offset_y = r * math.sin(math.radians(rotation_z_degrees))

    return offset_x, offset_y
'''


