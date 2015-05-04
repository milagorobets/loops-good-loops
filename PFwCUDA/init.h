/*
 * init.h
 *
 *  Created on: 2015-02-18
 *      Author: cinnamon
 */

#ifndef INIT_H_
#define INIT_H_

enum matrixFlow_types {BASIC, POSITIVE, NEGATIVE, RANDOM};
enum matrixWall_types {TENTH, ZERO, MIX};
enum source_types {SINE, IMPOSED_AT_START, CHANGED_ON_ITERATION};

void init_MatrixFlowType(matrixFlow_types* MATRIX_TYPE, float matrix[][4]);
void init_MatrixWallType(matrixWall_types* MATRIX_TYPE, float matrix[][4]);

#endif /* INIT_H_ */
