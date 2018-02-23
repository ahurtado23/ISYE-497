//Alexandra Hurtado
//Understanding Linear Algebra/Neural Network Mathmatical Model using Java


//var m = new Matrix(3,2); function generates a matrix
function Matrix(rows,cols) {


this.rows = rows;
this.cols = cols;
this.matrix = []; //creates an array to start matrix

//3 properties, # of rows, # of columns, and array attribute


//this is going to make the matrix because it loops thorugh rows and columns and defines them as arrays and makes them 0
for (var i = 0; i < this.rows; i++) {
	this.matrix[i] = [];
	for (var j = 0; j < this.cols; j++) {
		this.matrix[i][j] = 0;

	}
}

}

//function to randomize numbers

Matrix.prototype.randomize = function(n) { //this function adds x amount to the matrix
	for (var i = 0; i < this.rows; i++) {
		for (var j = 0; j < this.cols; j++) {
			this.matrix[i][j] = Math.floor(Math.random()*10);

		}

	}
}
		


Matrix.prototype.add = function(n) { //this function adds x amount to the matrix
	
if (n instanceof Matrix) { 
	for (var i = 0; i < this.rows; i++) {
		for (var j = 0; j < this.cols; j++) {
			this.matrix[i][j] += n.matrix[i][j];
		}
	}
} else {
	for (var i = 0; i < this.rows; i++) {
		for (var j = 0; j < this.cols; j++) {
			this.matrix[i][j] += n;

		}
	}
}
}



Matrix.prototype.multiply = function(n) { //this function multiples x amount to the matrix
	if (n instanceof Matrix) { //Matrix Product

		if (this.cols !== n.rows){
        console.log('Columns of A must equal columns of B')
			return undefined;
		}

		let a = this;
		let b = n;
  let result = new Matrix(a.rows, b.cols);


		for (let i = 0; i < result.rows; i++) {  //for each row
			for (let j = 0; j < result.cols; j++) { // for each column

//do dot product for all column spots and all row spots 

				let sum = 0;
				for (let k = 0; k < a.cols; k++) {
					sum += a.matrix[i][k] * b.matrix[k][j];
				}
 
result.matrix[i][j] = sum;

			}

		}


return result;

	} else { //Scalar Product

	for (let i = 0; i < this.rows; i++) {
		for (let j = 0; j < this.cols; j++) {
			this.matrix[i][j] *= n;

		}
	}

}
}



 