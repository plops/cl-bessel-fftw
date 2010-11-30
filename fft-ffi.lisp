(defpackage :fft
  (:use :cl :sb-alien :sb-c-call))
(in-package :fft)
(declaim (optimize (speed 2) (debug 3) (safety 3)))

(defconstant +forward+ 1)
(defconstant +backward+ -1)
(defconstant +measure+ 0)
(defconstant +estimate+ (ash 1 6)) ;; array isn't overwritten during planning

(load-shared-object "/usr/local/lib/libfftw3f.so.3")
(define-alien-type plan (* int))

;; power of two fastest; 2 3 5 7 11 13 should be the only dividors,
;; then fast algorithms are available; it is benificial if last
;; dimension of r2c/c2r transform should be even

(define-alien-routine fftwf_execute
    void
  (plan plan))

;; real input: n0 x n1 x n2 x .. x n_(d-1), d is the rank 
;; output; n0 x n1 x n2 x .. x (n_(d-1)/2+1)
;;                                        ^
;;                                        |
;; non-negative frequencies and one element 
;; array contents are overwritten during planning

;; for in-place transform real input in row-major order must be padded
;; two extra if last dimension is even and one if odd -> 2 (n_(d-1) /
;; 2 + 1) real values but only n_(d-1) values are stored

;; row-major: if you step through memory, the first dimension's index
;; varies most slowly: 

;; pos=i_(d-1) + nd (i_(d-2) + n_(d-2) (... + n1 i0))

;; the position (i,j,k) of a 5x12x27 array would be accessed with
;; array[k+27*(j+12*i)]

(define-alien-routine fftwf_plan_dft_r2c
    plan
  (rank int)
  (n (* int))
  (in (* single-float))
  (out (* single-float)) ;; actually complex
  (flags unsigned-int))

(define-alien-routine fftwf_plan_dft_c2r
    plan
  (rank int)
  (n (* int))
  (in (* single-float)) ;; actually complex
  (out (* single-float)) 
  (flags unsigned-int))

(load-shared-object "/usr/local/lib/libfftw3f_threads.so")
  
(define-alien-routine ("fftwf_init_threads" init-threads)
    int)

(define-alien-routine ("fftwf_plan_with_nthreads" plan-with-nthreads)
    void
  (nthreads int))

#+nil
(progn
 (init-threads)
 (plan-with-nthreads 2))


(defun plan (dims a)
  (declare ((simple-array single-float *) a))
  (let* ((rank (array-rank a))
	 (dims-a (make-array rank :element-type '(signed-byte 32)
			     :initial-contents dims))
	 (a-sap (sb-sys:vector-sap
		 (sb-ext:array-storage-vector a))))
    (sb-sys:with-pinned-objects (dims-a a)
      (fftwf_plan_dft_r2c rank (sb-sys:vector-sap dims-a)
			  a-sap a-sap +measure+))))

(defun ft (plan a)
  (declare ((simple-array single-float *) a))
  (sb-sys:with-pinned-objects (a)
    (fftwf_execute plan)))
#+nil
(let* ((n0 128)
       (n1 128)
       (n2 128)
       (n3 128)
       (a (make-array (list (* 2 (+ (floor n3 2) 1)) n2 n1 n0)
		      :element-type 'single-float))
       (plan (time (plan a))))
  (defparameter *plan* plan)
  (time (ft plan a))
  (aref a 0 0 0 0))


;; 1024x1024x512 transform takes 21s
;; 2048x2048x128 13s on one processor, 8s on two
;; 16192x32384 - takes too long to optimize
;; 8192x8192 2.5s on two, measure 117s
;; 8192x8192x2 6.4s on two, measure 200s (after 8192x8192)
;; 640X480X640 4s on two, measure 214s
;; 128x128x128x128 3.4s on two (really?), measure 89s 

(defun write-pgm (filename img)
  (declare (simple-string filename)
           ((array (unsigned-byte 8) 2) img)
           (values null &optional))
  (destructuring-bind (h w) (array-dimensions img)
    (declare ((integer 0 65535) w h))
    (with-open-file (s filename
                       :direction :output
                       :if-exists :supersede
                       :if-does-not-exist :create)
      (declare (stream s))
      (format s "P5~%~D ~D~%255~%" w h))
    (with-open-file (s filename 
                       :element-type '(unsigned-byte 8)
                       :direction :output
                       :if-exists :append)
      (let ((data1 (sb-ext:array-storage-vector img)))
        (write-sequence data1 s)))
    nil))

(defun draw-disk (img radius y x)
  (declare (single-float radius y x)
	   ((array single-float *) img)
	   (values (array single-float *) &optional))
  (destructuring-bind (h w) (array-dimensions img)
    (declare ((integer 0 65535) w h))
    (let ((rr (* radius radius))
	  (w2 (* .5 w))
	  (h2 (* .5 h)))
     (dotimes (j h)
       (dotimes (i w)
	 (let* ((x (- i x w2))
		(y (- j y h2))
		(r2 (+ (* x x) (* y y))))
	   (setf (aref img j i)
		 (if (< r2 rr)
		     1s0
		     0s0)))))))
  img)

(defun convert (img)
  (declare ((array single-float *) img))
  (let* ((img1 (make-array (array-total-size img) :element-type 'single-float
			   :displaced-to img))
	 (n (length img1))
	 (out (make-array (array-dimensions img)
			  :element-type '(unsigned-byte 8)))
	 (out1 (sb-ext:array-storage-vector out)))
    (dotimes (i n)
      (setf (aref out1 i) (floor (aref img1 i))))
    out))

(defun normalize (img)
  (declare ((array single-float *) img))
  (let* ((img1 (make-array (array-total-size img) :element-type 'single-float
			   :displaced-to img))
	 (mi (reduce #'min img1))
	 (ma (reduce #'max img1))
	 (s (/ 255s0 (- ma mi)))
	 (n (length img1)))
    (dotimes (i n)
      (setf (aref img1 i)
	    (* s (- (aref img1 i) mi))))
    img))

#+nil
(let* ((al (list 1s0 1.1s0 2s0 3s0))
       (a (make-array (length al)
		      :element-type 'single-float
		      :initial-contents al)))
  (normalize a))

(defconstant +pi+ (coerce pi 'single-float))

#+nil
(let* ((z 128)
       (w 128)
       (h 128)
       (vol (make-array (list (* 2 (+ (floor z 2) 1))
			      h w)
			 :element-type 'single-float)))
  (sb-sys:with-pinned-objects (vol)
    (let ((plan (time (plan vol))))
      (dotimes (i z)
	(let ((a (make-array (list h w)
			     :element-type 'single-float
			     :displaced-to vol
			     :displaced-index-offset (* w h i)))
	      (z (* 32s0 (exp (complex 0 (* 4s0 2s0 +pi+ i (/ z)))))))
 	  (draw-disk a 12s0 (realpart z) (imagpart z))))

      (time (ft plan vol))

      (dotimes (i (floor z 2))
	(let ((a (make-array (list h	(* 2 w))
			     :element-type 'single-float
			     :displaced-to vol
			     :displaced-index-offset (* 2 w h i)))
	      (mag (make-array (list h w)
			       :element-type 'single-float)))
	  (format t "~a~%" (list 'size (array-total-size mag)))
	  (dotimes (j h) 
	    (dotimes (i w)
	      (setf (aref mag j i) (abs (complex (aref a j i)
						 (aref a j (1+ (* 2 i))))))))
	  (normalize mag)  
	  (write-pgm (format nil "/dev/shm/o~3,'0d.pgm" i)
		     (convert mag)))))))


(let* ((w 128)
       (h 242)
       (a (make-array (list h (* 2 (1+ (floor w 2))))
		     :element-type 'single-float)))
  (draw-disk a 4.3s0 0s0 0s0)
  (ft (plan (list h w) a) a)
  (let ((b (make-array (list h
			     (1+ (floor w 2))
			     )
		       :element-type 'single-float)))
    (destructuring-bind (h w) (array-dimensions b)
      (dotimes (j h)
	(dotimes (i w)
	  (setf (aref b j i) (abs (complex (aref a j (* 2 i))
				       (aref a j (1+ (* 2 i)))))))))
    (normalize b)
    (write-pgm "/dev/shm/o.pgm" (convert b))))