import { describe, it, expect} from 'vitest'
import { reactive} from 'vue'

function makeLoginForm() {
  const form = reactive({ email: '', password: '' })
  const errors = reactive({ email: '', password: '' })

  function validate(): boolean {
    errors.email = ''
    errors.password = ''

    if (!form.email) {
      errors.email = 'Email is required'
    } else if (!/\S+@\S+\.\S+/.test(form.email)) {
      errors.email = 'Enter a valid email'
    }

    if (!form.password) {
      errors.password = 'Password is required'
    }

    return !errors.email && !errors.password
  }

  return { form, errors, validate }
}


describe('LoginView: validate()', () => {
  describe('email', () => {
    it('fails when email is empty', () => {
      const { form, errors, validate } = makeLoginForm()
      form.password = 'Secret1'

      expect(validate()).toBe(false)
      expect(errors.email).toBe('Email is required')
    })

    it('fails when email has no @', () => {
      const { form, errors, validate } = makeLoginForm()
      form.email = 'notanemail'
      form.password = 'Secret1'

      expect(validate()).toBe(false)
      expect(errors.email).toBe('Enter a valid email')
    })

    it('fails when email has no dot after @', () => {
      const { form, errors, validate } = makeLoginForm()
      form.email = 'user@nodot'
      form.password = 'Secret1'

      expect(validate()).toBe(false)
      expect(errors.email).toBe('Enter a valid email')
    })

    it('passes with valid email', () => {
      const { form, errors, validate } = makeLoginForm()
      form.email = 'user@example.com'
      form.password = 'Secret1'

      expect(validate()).toBe(true)
      expect(errors.email).toBe('')
    })
  })

  describe('password', () => {
    it('fails when password is empty', () => {
      const { form, errors, validate } = makeLoginForm()
      form.email = 'user@example.com'

      expect(validate()).toBe(false)
      expect(errors.password).toBe('Password is required')
    })

    it('passes with any non-empty password', () => {
      const { form, errors, validate } = makeLoginForm()
      form.email = 'user@example.com'
      form.password = 'x'

      expect(validate()).toBe(true)
      expect(errors.password).toBe('')
    })
  })

  describe('combined', () => {
    it('returns false and sets both errors when both fields are empty', () => {
      const { errors, validate } = makeLoginForm()

      expect(validate()).toBe(false)
      expect(errors.email).toBe('Email is required')
      expect(errors.password).toBe('Password is required')
    })

    it('clears previous errors on re-validation', () => {
      const { form, errors, validate } = makeLoginForm()

      validate()
      form.email = 'user@example.com'
      form.password = 'Secret1'
      validate()

      expect(errors.email).toBe('')
      expect(errors.password).toBe('')
    })

    it('returns true when both fields are valid', () => {
      const { form, validate } = makeLoginForm()
      form.email = 'admin@test.io'
      form.password = 'anypassword'

      expect(validate()).toBe(true)
    })
  })
})