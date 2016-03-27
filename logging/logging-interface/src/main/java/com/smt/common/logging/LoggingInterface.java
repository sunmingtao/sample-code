package com.smt.common.logging;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The logger instance is decided at runtime.
 * In this project, its implementation is slf4j-simple. 
 * In the client project, its implementation can be log4j or logback.
 * 
 * Run the unit test in logging-log4j project and logging-logback project
 * Note even though the same line of code is executed: logger.info("Hello World"),
 * the output is different between the 3 test cases.
 * @author Mingtao Sun
 */
public class LoggingInterface{
	/** Logger */
	private static final Logger logger = LoggerFactory.getLogger(LoggingInterface.class.getName());
	public void log(){
    	logger.info("Hello World");
    }
}
