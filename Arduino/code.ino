
#include <cvzone.h>
#include <Servo.h>
#include <string.h>

#include <stdint.h>

#include <LiquidCrystal_I2C.h>

LiquidCrystal_I2C lcd(0x27, 16, 2);

SerialData serialData(1, 1);

int valsRec[1];

//khai bao servo

Servo ServoCircle;

Servo ServoSquare;

Servo ServoTriangle;

//khai bao cam bien----------------

#define sensor1 7

#define sensor2 4

#define sensor3 2

#define sensor4 8

//mang nhan tin hieu tu Serial-----

int myArray[100];

int X = 0;

//

int in1 = 3;

int in2 = 5;

int reset = 13;

int gtreset;

int stopp = 12;

int counter1 = 0;

int counter2 = 0;

int counter3 = 0;

int counter4 = 0;

void setup() {

  ServoCircle.attach(11);

  ServoSquare.attach(9);

  ServoTriangle.attach(10);

  pinMode(in1, OUTPUT);

  pinMode(in2, INPUT);

  pinMode(reset, INPUT_PULLUP);

  pinMode(stopp, INPUT_PULLUP);

  ServoCircle.write(0);

  ServoSquare.write(36);

  ServoTriangle.write(3);

  serialData.begin(9600);

  lcd.begin(16, 2);

  lcd.backlight();

  lcd.setCursor(0, 0);

  lcd.print("Tr:");

  lcd.setCursor(8, 0);

  lcd.print("Vg:");

  lcd.setCursor(0, 1);

  lcd.print("T.g:");

  lcd.setCursor(10, 1);

  lcd.print("T:");
}

void resetBoard() {

  asm volatile("jmp 0");
}

void Motor() {

  digitalWrite(in1, HIGH);

  digitalWrite(in2, LOW);
}

void MotorOff() {

  digitalWrite(in1, LOW);

  digitalWrite(in2, LOW);
}

void loop() {


  Motor();

  if (digitalRead(stopp) == 0) {
    MotorOff();
  }

  if (digitalRead(reset) == 0) {
    resetBoard();
  }

  if (digitalRead(sensor4) == 0) {
    while (digitalRead(sensor4) == 0) {}
    counter4++;
  }

  if (Serial.available() > 0 ) {
    myArray[X] = Serial.parseInt();
    Serial.print(myArray[X]);
    while (myArray[X] == 1) {

      if (digitalRead(sensor1) == 0) {
        ServoCircle.write(55);
        delay(8000);
        ServoCircle.write(0);
        counter1++;

        break;
      }
    }

    while (myArray[X] == 2) {

      if (digitalRead(sensor2) == 0) {
        ServoSquare.write(100);
        delay(8000);
        ServoSquare.write(45);
        counter2++;
        break;
      }
    }

    while (myArray[X] == 3) {

      if (digitalRead(sensor3) == 0) {
        ServoTriangle.write(65);
        delay(8000);
        ServoTriangle.write(10);
        counter3++;
        break;
      }
    }
  }

  lcd.setCursor(3, 0);
  lcd.print(counter1);
  lcd.setCursor(11, 0);
  lcd.print(counter2);
  lcd.setCursor(4, 1);
  lcd.print(counter3);
  lcd.setCursor(12, 1);
  lcd.print(counter4);
  X++;
}
