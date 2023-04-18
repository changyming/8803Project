# Analysis of Covid-19 Vaccine Sentiment
- Yuming Chang


## Introduction/Summary
To analyze the sentiment of people's post on the social media is useful for understanding the opinion on crucial topics and providing customized service based on those information. It's impossible for people to do sentiment analysis by themselves, since the huge amount of data and bias from different person. In this project, I will introduce a method how to generate the sentiment based on the pure text using TextBlob and convert the text data into numeric matrix using TF-IDF method. After that, three machine learning and deep learning model are developed: Support Vector Machine (SVM), Convolutional Neural Network (CNN) and Graph Neural Network (GNN) to compared the performance of the three models on the sentiment prediction. And in this project, SVM model will serve as the baseline. The confusion matrix of each model also provided to see how each model performance on different class.

### Goals
My goals will be as following bullets:

- Clean the tweet data, such as remove the special marks.

- Generate the sentiment of the tweet text using TextBlob method.

- Generate the numerical matrix using TD-IDF

- Create SVM, CNN, CNN-LSTM models

- Tune Hyperparameters and Get the best models

- Compare the performance of each model and generate the confusion matrix of each model

## Results

### Progress step by step
Currently, we have basically two main tasks to solve:
1. To distinguish different scenarios: natural light, mixed(natrual and artificial lights), artificial light
2. To measure the comfortable light zone for the group members.
We have read several websites about how to implement and how to write the code to detect the lightness using photoresistor sensors and control the LEDs.

In order to accomplish these goals, we designed the experiments in the following.


- Step one: set up the hardware system. The following is the light measurement system:
<p align="center">
  <img src="https://github.com/YESAndy/12740-group-project/blob/gh-pages/lightmeasuresystem.jpg?raw=true">
</p>

<p align="center">
  Figure 1. measurement system
 </p>


- Step two: determine the light intensity ranges of natural light and artificial light separately
- Step three: determine the comfortable light ranges when doing step two

### coding progress
The following is part of the light measurement code:

<p align="center">
  <img src="https://github.com/YESAndy/12740-group-project/blob/gh-pages/lightmeasurecode.png?raw=true">
</p>

<p align="center">
  Figure 2. code progress
 </p>


### Problems Encountered
1.When we are going to implement the whole system using the code we wrote, we found there are some bugs in our code. And now we can't find where the bugs are.
<p align="center">
  <img src="https://github.com/YESAndy/12740-group-project/blob/main/lightmeasurebug.png?raw=true">
</p>

<p align="center">
  Figure 3. code problem
 </p>

2.Since we intended to do a lot of work off campus and remotely, we tried to connect the raspberry pi to share internet with our pcs and at-home wifi (no one has a monitor, keyboard, or required HDMI cable off-campus). Also the Cooperation between teammates is limited due to the remote study.

### Future Plan
Describe what you plan to do in the next two weeks
1. we are going to schedule serval meeting to talk more about the progress each one achived and how to do things better.
2. We are going to continue to debug our code to make the code and system can run successfully.
3. We are going to design a room, where we can test our detect system and code in serval situations.

## Methodology
### Phenomena of Interest

We want to measure the light intensity and corresponding temperature at the same time to see the pattern of light intensity and temperature.

__Visible Light__

Visible light is also part of the electromagnetic spectrum, at higher frequencies that IR radiation discussed previously. The wavelengths for visible light range between 400 -700 nm, and the most common measurement for visible light is its wavelength, in nanometers. All waves in the electromagnetic spectrum travel as the speed of light, commonly known as coefficient c.

Light has natural and artificial sources, including the Sun, fluorescent lamps, incandescent bulbs, LEDs and so on. Most light sources use the at least some part of the full spectrum of visible light but may be skewed towards particular wavelengths of light. Natural light does contain the full spectrum.[1]
 
 <p align="center">
  <img src="https://whybecausescience.files.wordpress.com/2014/01/visible-light-spectrum.jpg?w=1680">
</p>

<p align="center">
  Figure 4.  visible light spectrum[2]
 </p>



__Temperature__

Temperature is a physical quantity that expresses hot and cold. It is the manifestation of thermal energy, present in all matter, which is the source of the occurrence of heat, a flow of energy, when a body is in contact with another that is colder.

Temperature is measured with a thermometer. Thermometers are calibrated in various temperature scales that historically have used various reference points and thermometric substances for definition. The most common scales are the Celsius scale (formerly called centigrade, denoted °C), the Fahrenheit scale (denoted °F), and the Kelvin scale (denoted K), the last of which is predominantly used for scientific purposes by conventions of the International System of Units (SI).

The lowest theoretical temperature is absolute zero, at which no more thermal energy can be extracted from a body. Experimentally, it can only be approached very closely, but not reached, which is recognized in the third law of thermodynamics.[3]



__Infrared  Radiation__

Infrared or IR is a kind of radiation emitted by almost all objects, and is emitted as heat energy. It’s part of the electromagnetic (EM) spectrum, which also includes gamma-rays, x-rays, UV radiation and visible light. EM radiation is created by electric and magnetic oscillation fields perpendicular to each other as well as charged particles traveling in space.[4]

<p align="center">
  <img src="http://4.bp.blogspot.com/_ltlOkpoAR8o/S1osC5xABBI/AAAAAAAABJE/BtOTaWrDGSw/s400/500px-EM_spectrum.svg.png">
</p>

<p align="center">
  Figure 5.  light spectrum[5]
 </p>


### Sensor(s) Used
Describe the sensor(s) you used, e.g. physical principles, static and dynamic behavior, and signal characteristics

**MCP3008**

This is an Analog to Digital Converter

- 10-bit resolution

- 8 (MCP3008) input channels

- Single supply operation: 2.7V - 5.5V

- Industrial temp range: -40°C to +85°C

<p align="center">
  <img src="https://www.microchip.com/_images/ics/medium-MCP3008-PDIP-16.png">
</p>

<p align="center">
  Figure 6.  MCP3008
 </p>

**Photosensitive Light Sensor Module**

- Measure the light level at the bottom

- Input Voltage: 3.3V to 5V

- Signal output indicator light

- LDR module 4 PIN

- Able to detect ambient brightness and light intensity Adjustable sensitivity

<p align="center">
  <img src="https://leobot.net/productimages/222.jpg">
</p>

<p align="center">
  Figure 7.  Photosensitive Light Sensor Module
 </p>


**DHT11 Temperature and Humidity Sensor Module**

- Accuracy:±5%

- Humidity Range:20 ~ 80% RH

- Mounting Type:Through Hole

- Operating Temperature:0°C ~ 50°C

- Output:16b

- Output Type:Digital

- Package / Case:Housed Sensor

- Response Time:1s

- Sensor Type:Humidity, Temperature

- Voltage - Supply:3V ~ 5V

<p align="center">
  <img src="https://leobot.net/productimages/1631.jpg">
</p>

<p align="center">
  Figure 8.  Photosensitive Light Sensor Module
 </p>


**HC-SR501 Infrared PIR Motion Sensor Module**

- Voltage: 5V – 20V

- Power Consumption: 65mA

- TTL output: 3.3V, 0V

- Trigger methods: L – disable repeat trigger, H enable repeat trigger

- Sensing range: less than 120 degree, within 7 meters

- Temperature: – 15 ~ +70

- Dimension: 32*24 mm, distance between screw 28mm, M2, Lens dimension in diameter: 23mm

<p align="center">
  <img src="https://periph.io/img/hc-sr501.jpg">
</p>

<p align="center">
  Figure 9.  Photosensitive Light Sensor Module
 </p>




### Physical Principle

**MCP3008**

MCP3008 is an eight channel, 10-bit ADC SAR ADC, which is a successive approximation analog-to-digital converter. The “eight channels” refers to the fact that the MCP3008 is able to receive and process voltages on eight different inputs, number 0-7. These channels can be read in two different ways. The typical fashion is called “single ended input”, where the voltage level is read between the selected voltage reference, VREF, easily selected on the I2C and SPI Education Shield using the VREF jumper, and a ground level shared in common with the circuit doing the measuring and the circuit being measured. The 10 bits refers to the sampling range of the ADC itself. B0000000000 = 0 and B1111111111 = 1023, meaning you can measure 1024 different voltage levels, typically between ground and VREF. So if you have a 5V reference, and can split it 1024 times, each step would be equivalent to a measurement of 0.0049V. With the 3V3 reference, each step is equivalent to a measurement of 0.0032V. [6]

<p align="center">
  <img src="https://cdn-learn.adafruit.com/assets/assets/000/030/456/original/sensors_raspberry_pi_mcp3008pin.gif?1455010861">
</p>

<p align="center">
  Figure 10.  MCP3008[7]
 </p>



**Photosensitive Light Sensor Module**

The photoresistor is based on the internal photoelectric effect. Photosensitive resistors are formed by mounting electrode leads at both ends of the semiconductor photosensitive material and encapsulating them in a tube case with a transparent window. In order to increase the sensitivity, the two electrodes are often made into a comb shape. The materials used to make photoresistors are mainly semiconductors such as metal sulfides, selenides, and tellurides. Coating, spraying, sintering and other methods are used to make a very thin photoresistor and a comb-shaped ohmic electrode on an insulating substrate. The leads are connected and sealed in a sealed housing with a light-transmitting mirror to prevent its sensitivity from being affected by moisture. After the incident light disappears, the electron-hole pairs generated by the photon excitation will recombine, and the resistance of the photoresistor will return to its original value. When a voltage is applied to the metal electrodes at both ends of the photoresistor, a current passes through it. When the photoresistor is irradiated by the light with a certain wavelength, the current will increase with the light intensity, thereby achieving photoelectric conversion. The photoresistor has no polarity and is purely a resistive device. It can be used with both DC voltage and AC voltage. The conductivity of a semiconductor depends on the number of carriers in the semiconductor's conduction band.[8]

<p align="center">
  <img src="https://api.utmel.com/Upload/Images/Article/1a00ebb5-f5f5-4ac7-ba9f-bfc1774cdd50.jpg">
</p>

<p align="center">
  Figure 11. Photosensitive Light Sensor Module[9]
 </p>


**DHT11 Temperature and Humidity Sensor Module**
<p align="center">
  <img src="https://howtomechatronics.com/wp-content/uploads/2016/01/DHT11-DDHT22-Working-Principle.png">
</p>

<p align="center">
  Figure 12. DHT11 Temperature and Humidity Sensor Module[10]
 </p>

For measuring humidity they use the humidity sensing component which has two electrodes with moisture holding substrate between them. Between the electrodes, there is a moisture-holding substrate that can absorb water vapor. The substrate releases free ions, which increases the conductivity between the electrode, as water vapor enters it. Thus, the humidity sensing component is a moisture holding substrate with electrodes applied to the surface. When water vapor is absorbed by the substrate, ions are released by the substrate which increases the conductivity between the electrodes. The change in resistance between the two electrodes is proportional to the relative humidity. Higher relative humidity decreases the resistance between the electrodes, while lower relative humidity increases the resistance between the electrodes[11]




<p align="center">
  <img src="https://howtomechatronics.com/wp-content/uploads/2016/01/Humidity-Sensor-Working-Principle.jpg">
</p>

<p align="center">
  Figure 13. Humidity sensor work principles[12]
 </p>


A thermistor is actually a variable resistor that changes its resistance with change of the temperature. These sensors are made by sintering of semiconductive materials such as ceramics or polymers in order to provide larger changes in the resistance with just small changes in temperature. The term “NTC” means “Negative Temperature Coefficient”, which means that the resistance decreases with increase of the temperature.[13]

<p align="center">
  <img src="https://howtomechatronics.com/wp-content/uploads/2016/01/Thermistor-Working-Principle.jpg">
</p>

<p align="center">
  Figure 14. Temperature sensor work principles[13]
 </p>


Negative temperature coefficient of resistance thermistors, or NTC thermistors for short, reduce or decrease their resistive value as the operating temperature around them increases. Generally, NTC thermistors are the most commonly used type of temperature sensors as they can be used in virtually any type of equipment where temperature plays a role. NTC temperature thermistors have a negative electrical resistance versus temperature (R/T) relationship. The relatively large negative response of an NTC thermistor means that even small changes in temperature can cause significant changes in their electrical resistance. This makes them ideal for accurate temperature measurement and control.[14]

Another important characteristic of a thermistor is its “B” value.  B value will define the thermistors resistive value at a first temperature or base point, called T1, and the thermistors resistive value at a second temperature point.[15]


**HC-SR501 Infrared PIR Motion Sensor Module**

PIR sensor is specially designed to detect such levels of infrared radiation. It basically consists of two main parts: A Pyroelectric Sensor and A special lens called Fresnel lens which focuses the infrared signals onto the pyroelectric sensor.

<p align="center">
  <img src="https://lastminuteengineers.com/wp-content/uploads/arduino/PIR-Sensor-Working-Pyroelectric-Sensor-Two-Detection-Slots.png">
</p>

<p align="center">
  Figure 15. Temperature sensor work principles[16]
 </p>


There are two potentiometers on the back of the PIR chip to control the sensitivity of the PIR and the delay time. The sensitivity of the PIR sensor is the sensing capability; the sensor can detect anywhere from between 3 to 7 meters, adjusted by the potentiometer. Delay time is the amount of time that the PIR sensor will remain high after being triggered by motion. It can help reduce redundant readings when counting fast-moving objects and provides flexibility in sensor setup. The delay time can be set to read high for 3 seconds to 5 minutes, and is always followed by 3 seconds of low output where no readings can be made. Another feature of the PIR sensor is the Trigger Mode, which has a Repeatable (H mode) and Single (L mode) option. The single trigger option means that the time delay begins when the motion is first detected, and ignores any additional triggers while remaining high for the rest of the delay time. Repeatable triggers mean that ever motion detection will reset the sensor’s delay time to restart count[17]. 



### Signal Conditioning and Processing
__Decision making__

The decision making for the automatic adjustment of light intensity is based on the scenarios that the system detects. Here we have 3 states of interests: No led opening, One LED opening, and Two LED opening. To make a smarter decision, we make the following decision making workflow:

```pseudocode
#   
for every sampling:

    human = pir.result
    current_light_intensity = light_sensor.result
    
    if there exists human in the room:
        
        if current light is too bright:
            if there is still openning LED
                close one LED
        else if current light is too dark:
            if the # of LED is less than 2:
                open one LED
    else:
    
        close all LED
```

The workflow above is simple and intuitive. However, through several tests, we found that the human detection part is not stable. That is, even if there is continuously something warm in front of the PIR sensor, it still produces some "FALSE" feedback. The reasons why this happens may be: 1) the measure range of PIR is not enough; 2) the connection between PIR and the control system is not stable

We then found that even if one single signal is not stable, a series of signals will produce a significant portion of right feedback. Based on this finding, we develop a context-based signal prediction algorithm:
<p align="center">
  <img src="https://github.com/YESAndy/12740-group-project/blob/main/contextalgorithm.png?raw=true">
</p>

<p align="center">
  Figure 16. context algorithm
 </p>


## Experiments and Results
### Implementation of the context-based human detection algorithm
The distribution of the portion samples

<p align="center">
  <img src="https://github.com/YESAndy/12740-group-project/blob/main/portiondistribution.png?raw=true">
</p>

<p align="center">
  Figure 17. Portion distribution
 </p>

Result of the confidence interval is

- (0.566, 0.847)

Note that we only use the minimum of interval as the threshold.

### The toy model
We set up a toy model to simulate the real environment (Figure 18). The hardware system is on the top of the model. And there is a tiny tiger sitting inside the model as our 
tested objectives

<p align="center">
  <img src="https://github.com/YESAndy/12740-group-project/blob/main/IMG_2038.JPG?raw=true">
</p>

<p align="center">
  Figure 18. Experiment environment
 </p>

